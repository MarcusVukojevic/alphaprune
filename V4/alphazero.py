import os
import math
import random
import torch
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

from mcts import MCTS


class AlphaZero:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game          # PruneGame condiviso (modello + cache)
        self.args = args

        self.mcts = MCTS(game, model, args)   # usato nel percorso sequenziale
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

        # Parallel
        self.parallel = bool(args.get("parallel", False))
        self.parallel_workers = int(args.get("parallel_workers", 0))

    # --------------------
    # Percorso SEQUENZIALE
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

    # --------------------
    # Percorso PARALLELO
    # --------------------
    class _EpisodeEnv:
        """
        Env leggero per episodio, non tocca lo stato interno del PruneGame.
        Delegando la metrica a core.evaluate_metric_shared(..., stage_idx),
        più episodi possono coesistere in parallelo condividendo il modello.
        """
        def __init__(self, core_game, args):
            from collections import deque as _deque
            self.core = core_game
            self.args = args
            self.device = core_game.device

            self.target_sparsity = core_game.target_sparsity
            self.limite_mosse = core_game.limite_mosse
            self.eval_mode = core_game.eval_mode

            self.ppl_tol_frac = args.get("ppl_tol_frac", 0.10)
            self.ppl_alpha    = args.get("ppl_alpha", 0.05)
            self.kl_alpha     = args.get("kl_alpha", 0.02)
            self.kl_tol       = args.get("kl_tol", 0.05)
            self.mse_alpha    = args.get("mse_alpha", 0.02)
            self.mse_tol      = args.get("mse_tol", 0.02)
            self.require_metric_ok_for_stop = args.get("require_metric_ok_for_stop", False)
            self.s_beta = args.get("s_beta", 0.05)
            self.eps_sparsity_bonus = args.get("eps_sparsity_bonus", 0.2)

            # stato locale
            self.state = core_game.initial_state.clone().to(self.device)
            self.initial_state = self.state.clone()
            self.history = _deque(maxlen=self.limite_mosse - 1)
            self.history.appendleft(self.initial_state)
            self.reward_history = []
            self.last_real_action = None
            self.numero_mossa = 0

            # logging metrica corrente (solo per info)
            self.ppl = getattr(core_game, "initial_ppl", None)
            self.kl = 0.0 if self.eval_mode == "kl" else None
            self.mse = 0.0 if self.eval_mode == "mse" else None

        def reset_game(self):
            self.state.copy_(self.initial_state)
            self.numero_mossa = 0
            self.history.clear()
            self.history.appendleft(self.initial_state)
            self.reward_history.clear()
            self.last_real_action = None
            return self.state

        def get_encoded_state(self, state: torch.Tensor):
            T, N = self.limite_mosse, state.numel()
            enc = torch.zeros((T, N), dtype=torch.float32, device=self.device)
            enc[0] = state.float()
            for t, past_state in enumerate(self.history, start=1):
                if t >= T:
                    break
                enc[t] = past_state.float()
            return enc

        def get_scalar(self):
            return torch.tensor([self.limite_mosse - self.numero_mossa],
                                dtype=torch.float32, device=self.device)

        def legal_mask(self, state, is_root: bool, parent_action: int | None):
            mask = state.bool().clone()
            if is_root and self.last_real_action is not None:
                mask[self.last_real_action] = False
            if parent_action is not None:
                mask[parent_action] = False
            return mask

        def pensa_azione(self, state: torch.Tensor, action: int):
            if int(state[action].item()) == 0:
                return state
            s2 = state.clone()
            s2[action] = 0
            return s2

        # ---------- Surrogato progressivo ----------
        @torch.no_grad()
        def metric_progressive(self, state: torch.Tensor, stage_idx: int | None):
            return self.core.evaluate_metric_shared(state, stage_idx=stage_idx)

        def get_stage_baseline_for_ppl(self, stage_idx: int):
            return self.core.get_stage_baseline_for_ppl(stage_idx)

        @torch.no_grad()
        def get_value_and_terminated(self, state, depth=None, register=False):
            # NB: qui lasciamo il check su metrica "full" per stabilità del terminal check
            steps = self.numero_mossa if depth is None else depth
            s = 1.0 - state.float().mean().item()
            target = self.target_sparsity

            # valutazione completa (non surrogata) per la condizione di stop
            m_now = self.core.metric_from_state_or_cache(state)

            if self.eval_mode == "ppl":
                init_ppl = self.core.initial_ppl
                rel = max(0.0, (m_now - init_ppl) / (init_ppl + 1e-8))
                m_term = math.exp(- rel / (self.ppl_alpha + 1e-12))
                m_ok = (rel <= self.ppl_tol_frac)
            elif self.eval_mode == "kl":
                m_term = math.exp(- m_now / (self.kl_alpha + 1e-12))
                m_ok = (m_now <= self.kl_tol)
            else:
                m_term = math.exp(- m_now / (self.mse_alpha + 1e-12))
                m_ok = (m_now <= self.mse_tol)

            hit_s = (s >= target)
            hit_q = m_ok if self.require_metric_ok_for_stop else True
            limit = (steps >= self.limite_mosse)

            if hit_s and hit_q:
                reward = min(1.0, 0.7 + 0.3 * m_term)
                if register:
                    self.reward_history.append(float(reward))
                return float(reward), True

            if not limit:
                return 0.0, False

            progress = max(0.0, min(1.0, s / max(target, 1e-8)))
            reward = 0.4 * (progress * (0.5 + 0.5 * m_term))
            if register:
                self.reward_history.append(float(reward))
            return float(reward), True

        def do_action(self, gid: int):
            if int(self.state[gid].item()) == 0:
                self.numero_mossa += 1
                self.last_real_action = gid
                self.history.appendleft(self.state.clone())
                return self.state
            self.state[gid] = 0
            self.numero_mossa += 1
            self.last_real_action = gid
            self.history.appendleft(self.state.clone())
            # logging only: non necessario ai fini della policy
            m = self.core.metric_from_state_or_cache(self.state)
            if self.eval_mode == "ppl":
                self.ppl = m
            elif self.eval_mode == "kl":
                self.kl = m
            else:
                self.mse = m
            return self.state

    @torch.no_grad()
    def self_play_parallel(self):
        """
        Lancia num_self_iteration episodi in parallelo usando il modello vittima
        e la cache del PruneGame condivisi.
        """
        episodes = self.num_self_iteration
        results = []

        if self.parallel_workers <= 1:
            # fallback sequenziale (riusa self_play classico)
            for _ in range(episodes):
                results.append(self.self_play())
            return results

        def _run_one():
            env = AlphaZero._EpisodeEnv(self.game, self.args)
            mcts = MCTS(env, self.model, self.args)
            traj = []
            _ = env.reset_game()
            mcts.last_root = None
            while True:
                st = env.state.clone()
                action = mcts.search(st)
                root = mcts.last_root
                N = st.numel()
                pi = torch.zeros(N, dtype=torch.float32)
                tot = sum(c.visit_count for c in root.children)
                if tot > 0:
                    for c in root.children:
                        pi[c.action_taken] = c.visit_count / tot
                enc = env.get_encoded_state(st).cpu()
                scal = env.get_scalar().cpu().unsqueeze(0)
                env.do_action(action)
                reward, done = env.get_value_and_terminated(env.state, depth=env.numero_mossa, register=True)
                traj.append((enc, scal, pi.cpu(), None))
                if done:
                    final_ret = torch.tensor([reward], dtype=torch.float32)
                    traj = [(e, s, p, final_ret) for (e, s, p, _) in traj]
                    break
            return traj

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as ex:
            futs = [ex.submit(_run_one) for _ in range(episodes)]
            for f in as_completed(futs):
                results.append(f.result())
        return results

    # --------------------
    # Training
    # --------------------
    def train_on_memory(self, batch):
        device = next(self.model.parameters()).device
        states, scalars, pis, rets = zip(*batch)
        xx = torch.stack(states).to(device)   # (B,T,N)
        ss = torch.stack(scalars).to(device)  # (B,1)
        aa = torch.stack(pis).to(device)      # (B,N)
        rr = torch.stack(rets).to(device)     # (B,1)

        loss, pol, val, ent = self.model.fwd_train(xx, ss, aa, rr, lambda_H=self.lambda_H)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def learn(self):
        os.makedirs("models", exist_ok=True)

        for it in trange(self.num_episodes, desc="Iterations"):
            self.model.eval()

            # --- self-play (parallelo o sequenziale) ---
            if self.parallel:
                episodes = self.self_play_parallel()
                for ep in episodes:
                    self.replay.extend(ep)
            else:
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
