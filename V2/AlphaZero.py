# alphazero.py
import os
import random
import torch
from torch.nn.utils import clip_grad_norm_
from mcts import MCTS
from tqdm import trange

class AlphaZero:
    def __init__(self, model, game, args):
        self.model = model
        self.game  = game
        self.args  = args
        self.mcts = MCTS(game, model, args)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])

        # Hyper
        self.num_self_iteration = args["num_self_iteration"]   # quante partite per iterazione
        self.num_episodes = args["num_episodes"]         # quante iterazioni totali
        self.num_epochs = args["num_epochs"]           # quante epoche di training per iterazione
        self.batch_size = args["batch_size"]
        self.grad_clip = args["grad_clip"]
        self.lambda_H = args["entropy_bonus"]

        # Replay ugabuga
        self.replay = []

    
    @torch.no_grad()
    def self_play(self):

        traj = []
        state = self.game.reset_game()

        while True:
            # 1) MCTS
            action = self.mcts.search(state)

            # 2) Costruisci π dalle visite dei figli della root
            root = self.mcts.last_root
            N = state.numel()
            pi = torch.zeros(N, dtype=torch.float32)
            total_visits = 0
            for child in root.children:
                pi[child.action_taken] = child.visit_count
                total_visits += child.visit_count
            if total_visits > 0:
                pi /= total_visits

            # 3) Encoded state & scalar
            enc  = self.game.get_encoded_state(state).cpu()
            scal = self.game.get_scalar().cpu().unsqueeze(0)  # (1,)

            # 4) Esegui l'azione vera
            self.game.do_action(action)
            # 5) Check fine
            reward, done = self.game.get_value_and_terminated(self.game.state, depth=self.game.numero_mossa)
            traj.append((enc, scal, pi.cpu(), None))

            if done:
                final_ret = torch.tensor([reward], dtype=torch.float32)
                traj = [(e, s, p, final_ret) for (e, s, p, _) in traj]
                break

        return traj

    
    def train_on_memory(self, batch):

        device = next(self.model.parameters()).device

        states, scalars, pis, rets = zip(*batch)
        xx = torch.stack(states).to(device)    # (B,T,N)
        ss = torch.stack(scalars).to(device)   # (B,1)
        aa = torch.stack(pis).to(device)       # (B,N)
        rr = torch.stack(rets).to(device)      # (B,1) o (B,)

        loss, pol, val, ent = self.model.fwd_train(xx, ss, aa, rr, lambda_H=self.lambda_H)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    
    def learn(self):
        os.makedirs("models", exist_ok=True)
        import time


        # Iterazioni globali
        for it in trange(self.num_episodes, desc="Iterations"):
            self.model.eval()
            start = time.time()
            for _ in trange(self.num_self_iteration, desc=f"Self-play {it}", leave=False):
                episode = self.self_play()
                self.replay.extend(episode)
            finish = time.time()
            print(f"Ci mette: {finish-start}")
            if not self.replay:
                print("[warn] replay vuota, skip training")
                continue

            self.model.train()
            for ep in trange(self.num_epochs, desc=f"Train {it}", leave=False):
                batch = random.sample(self.replay, k=min(self.batch_size, len(self.replay)))
                self.train_on_memory(batch)

            # -------- SAVE ---------
            torch.save(self.model.state_dict(), f"models/model_iter{it}.pt")
