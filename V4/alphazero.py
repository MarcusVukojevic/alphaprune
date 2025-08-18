import os
import random
import threading
import torch
from torch.nn.utils import clip_grad_norm_
from mcts import MCTS
from collections import deque, OrderedDict
from tqdm import trange


class ThreadSafeLRU:
    def __init__(self, capacity: int = 20000):
        self._cap = int(capacity)
        self._od = OrderedDict()
        self._lock = threading.RLock()

    def get(self, k):
        with self._lock:
            if k in self._od:
                v = self._od.pop(k)
                self._od[k] = v
                return v
            return None

    def put(self, k, v: float):
        with self._lock:
            self._od[k] = float(v)
            if len(self._od) > self._cap:
                self._od.popitem(last=False)

    def clear(self):
        with self._lock:
            self._od.clear()


class AlphaZero:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        self.mcts = MCTS(game, model, args)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])

        # Hyper
        self.num_self_iteration = args["num_self_iteration"]  # partite per iterazione
        self.num_episodes = args["num_episodes"]              # iterazioni totali
        self.num_epochs = args["num_epochs"]                  # epoche di training per iterazione
        self.batch_size = args["batch_size"]
        self.grad_clip = args["grad_clip"]
        self.lambda_H = args["entropy_bonus"]

        # Replay buffer con cap (evita diluizione di esempi vecchi)
        cap = args.get("replay_maxlen", 10000)
        self.replay = deque(maxlen=cap)

    # --------------------
    # Percorso SEQUENZIALE (basic)
    # --------------------
    @torch.no_grad()
    def self_play(self):
        traj = []
        _ = self.game.reset_game()
        if hasattr(self.game, "clear_ppl_cache"):
            self.game.clear_ppl_cache()
        self.mcts.last_root = None

        while True:
            # stato corrente PRIMA dell'azione
            state = self.game.state.clone()

            # pianifica da questo stato con MCTS
            action = self.mcts.search(state)
            root = self.mcts.last_root

            # π dai visit count alla radice
            N = state.numel()
            pi = torch.zeros(N, dtype=torch.float32)
            tot = sum(c.visit_count for c in root.children)
            if tot > 0:
                for c in root.children:
                    pi[c.action_taken] = c.visit_count / tot

            # features pre-azione
            enc = self.game.get_encoded_state(state).cpu()
            scal = self.game.get_scalar().cpu().unsqueeze(0)

            # esegui l'azione e riusa il sottoalbero
            self.game.do_action(action)
            if hasattr(self.mcts, "reuse_subtree"):
                self.mcts.reuse_subtree(action)

            # check fine episodio
            reward, done = self.game.get_value_and_terminated(
                self.game.state, depth=self.game.numero_mossa, register=True
            )
            traj.append((enc, scal, pi.cpu(), None))

            if done:
                final_ret = torch.tensor([reward], dtype=torch.float32)
                traj = [(e, s, p, final_ret) for (e, s, p, _) in traj]
                break

        return traj

    def train_on_memory(self, batch):
        device = next(self.model.parameters()).device

        states, scalars, pis, rets = zip(*batch)
        xx = torch.stack(states).to(device)  # (B,T,N)
        ss = torch.stack(scalars).to(device)  # (B,1)
        aa = torch.stack(pis).to(device)  # (B,N)
        rr = torch.stack(rets).to(device)  # (B,1)

        loss, pol, val, ent = self.model.fwd_train(xx, ss, aa, rr, lambda_H=self.lambda_H)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def learn(self):
        os.makedirs("models", exist_ok=True)

        for it in trange(self.num_episodes, desc="Iterations"):
            self.model.eval()
            for _ in trange(self.num_self_iteration, desc=f"Self-play {it}", leave=False):
                episode = self.self_play()
                self.replay.extend(episode)
            if not self.replay:
                print("[warn] replay vuota, skip training")
                continue

            self.model.train()
            for _ in trange(self.num_epochs, desc=f"Train {it}", leave=False):
                batch = random.sample(list(self.replay), k=min(self.batch_size, len(self.replay)))
                self.train_on_memory(batch)

            torch.save(self.model.state_dict(), f"models/model_iter{it}.pt")

    # --------------------
    # PARALLEL LEARN (una sola GPU): più istanze di PruneGame sullo stesso device
    # con cache metrica condivisa; training centralizzato su self.model
    # --------------------
    @torch.no_grad()
    def _run_episode_on(self, game, policy_model):
        """Episodio singolo su una istanza di PruneGame, usando un MCTS locale."""
        mcts = MCTS(game, policy_model, self.args)
        traj = []
        _ = game.reset_game()
        mcts.last_root = None

        while True:
            st = game.state.clone()
            action = mcts.search(st)
            root = mcts.last_root

            N = st.numel()
            pi = torch.zeros(N, dtype=torch.float32)
            tot = sum(c.visit_count for c in root.children)
            if tot > 0:
                for c in root.children:
                    pi[c.action_taken] = c.visit_count / tot

            enc = game.get_encoded_state(st).cpu()
            scal = game.get_scalar().cpu().unsqueeze(0)

            game.do_action(action)
            reward, done = game.get_value_and_terminated(game.state, depth=game.numero_mossa, register=True)
            traj.append((enc, scal, pi.cpu(), None))
            if done:
                final_ret = torch.tensor([reward], dtype=torch.float32)
                traj = [(e, s, p, final_ret) for (e, s, p, _) in traj]
                break
        return traj

    def parallel_learn(self):
        """
        Esegue self-play parallelo creando N istanze di PruneGame *sullo stesso device*
        (es. 'cuda'), tutte con una cache LRU condivisa per le metriche.
        Ogni istanza esegue episodi in un thread; al termine si fa il training centrale.
        """
        os.makedirs("models", exist_ok=True)

        # device unico (stessa GPU o CPU) da args
        dev = self.args["device"]

        # quante istanze parallele (tutte sullo stesso device)
        env_count = max(1, int(self.args.get("parallel_workers", 2)))

        # cache metrica condivisa tra tutte le istanze
        shared_cache = ThreadSafeLRU(capacity=self.args.get("metric_cache_cap", 20000))

        # dataset comune (dalla istanza base già creata fuori)
        base_dataset = getattr(self.game, "dataset", None)
        if base_dataset is None:
            raise RuntimeError("PruneGame base non inizializzato correttamente (dataset mancante).")

        from prune_game import PruneGame
        from model import PruneModel

        for it in trange(self.num_episodes, desc="Iterations"):
            # distribuisci le partite tra le istanze
            per_env = [self.num_self_iteration // env_count] * env_count
            for i in range(self.num_self_iteration % env_count):
                per_env[i] += 1

            # snapshot dei pesi della policy
            policy_sd = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

            # costruisci istanze (tutte su dev)
            workers = []
            for idx in range(env_count):
                if per_env[idx] == 0:
                    continue
                child_args = dict(self.args)
                child_args["device"] = dev
                child_args["shared_metric_cache"] = shared_cache
                child_args["external_dataset"] = base_dataset  # stessi sample per tutti

                game_i = PruneGame(child_args)

                # policy locale (solo inferenza)
                num_blocks = game_i.initial_state.numel()
                history_len = self.args["n_mosse_massimo"]
                policy_i = PruneModel(num_blocks, history_len).to(dev)
                policy_i.load_state_dict(policy_sd, strict=True)
                policy_i.eval()

                workers.append((per_env[idx], game_i, policy_i))

            # lancia i thread
            episodes_collected = []
            threads = []
            results = [None] * len(workers)

            def _worker(idx, quota, g_i, p_i):
                local_eps = []
                for _ in range(quota):
                    ep = self._run_episode_on(g_i, p_i)
                    local_eps.extend(ep)
                results[idx] = local_eps

            for idx, (quota, g_i, p_i) in enumerate(workers):
                th = threading.Thread(target=_worker, args=(idx, quota, g_i, p_i), daemon=True)
                th.start()
                threads.append(th)
            for th in threads:
                th.join()

            for r in results:
                if r:
                    episodes_collected.extend(r)

            if episodes_collected:
                self.replay.extend(episodes_collected)
            else:
                print("[warn] nessun episodio raccolto in questa iter parallela")

            if not self.replay:
                print("[warn] replay vuota, skip training")
                continue

            self.model.train()
            for _ in trange(self.num_epochs, desc=f"Train {it}", leave=False):
                batch = random.sample(list(self.replay), k=min(self.batch_size, len(self.replay)))
                self.train_on_memory(batch)

            torch.save(self.model.state_dict(), f"models/model_iter{it}.pt")
