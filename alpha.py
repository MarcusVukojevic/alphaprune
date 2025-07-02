import os
import random
import math
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from collections import deque
from mcts import MCTS


class AlphaZero:
    """Minimal AlphaZero-style trainer for PruneGame, now with
    – online reward normalisation (Welford)
    – running loss meter (policy / value / total)
    – automatic PNG plot of the loss curves at the end of training
    """
    def __init__(self, model, optimizer, game, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

        # ----------------- running statistics for reward ------------------
        self.ret_mean = 0.0
        self.ret_var = 1.0  # cumulative variance * (n‑1)
        self.ret_count = 1e-4  # to avoid division by zero

        # ----------------- loss tracking ----------------------------------
        self.curve = []  # [(iter, ep, pol, val, tot), ...]
        self._reset_loss_meter()

        self.replay = deque(maxlen=self.args.get("replay_size", 1000))

    # ====================== helpers: reward stats =========================
    def _update_ret_stats(self, returns: torch.Tensor):
        """Welford online variance update"""
        for r in returns.view(-1):
            self.ret_count += 1.0
            delta = r.item() - self.ret_mean
            self.ret_mean += delta / self.ret_count
            self.ret_var += delta * (r.item() - self.ret_mean)

    def _ret_std(self):
        return math.sqrt(max(self.ret_var / (self.ret_count - 1), 1e-6))

    # ====================== helpers: loss meter ===========================
    def _reset_loss_meter(self):
        self._loss_pol, self._loss_val = [], []

    def _log_batch_loss(self, pol_loss, val_loss, entr_loss):
        self._loss_pol.append(pol_loss.item())
        self._loss_val.append(val_loss.item())

    def _flush_loss_meter(self, iter_id: int, ep_id: int):
        if not self._loss_pol:
            return
        m_pol = sum(self._loss_pol) / len(self._loss_pol)
        m_val = sum(self._loss_val) / len(self._loss_val)
        m_tot = m_pol + m_val
        print(f"[Iter {iter_id}  Ep {ep_id}]  pol={m_pol:.4f}  val={m_val:.4f}  tot={m_tot:.4f}")
        self.curve.append((iter_id, ep_id, m_pol, m_val, m_tot))
        self._reset_loss_meter()

    @torch.no_grad()
    def self_play(self):
        """
        Esegue un episodio completo di self-play:
        • ad ogni stato usa MCTS per scegliere l’azione
        • salva (encoded_state, scalar, π) prima di muovere
        • applica l’azione all’ambiente
        • al termine propaga lo stesso reward finale a tutti i frame
        Ritorna:
            List[Tuple[Tensor, Tensor, Tensor, Tensor]]
            con shape  (T,N) , (1,) , (N*ops,) , (1,)
        """
        trajectory = []                              # [(enc, scal, π), …]
        state      = self.game.get_initial_state()

        while True:
            # -------- 1) ricerca e scelta azione ----------------------------
            action = self.mcts.search(state)         # tensor([gate_id, op])

            # -------- 2) log root policy π ---------------------------------
            enc  = self.game.get_encoded_state(state)   # (T, N)
            scal = self.game.get_scalar()               # (1,)

            N_ops  = self.model.num_ops * enc.size(-1)  # N*ops
            visits = torch.zeros(N_ops, device=enc.device)
            for child in self.mcts.last_root.children:
                b, o = child.action_taken               # gate_id, op
                visits[b * self.model.num_ops + o] = child.visit_count
            pi = visits / visits.sum()                  # (N*ops,)

            trajectory.append((enc.cpu(), scal.cpu(), pi.cpu()))

            # -------- 3) esegui l’azione nell’ambiente ---------------------
            state  = self.game.perform_action(action)
            reward, done = self.game.get_value_and_terminated(state)
            if done:
                break

        final_r = torch.tensor([reward])               # (1,)
        # broadcast del reward finale a tutti i frame
        return [(st, sc, pi, final_r) for st, sc, pi in trajectory]


    def train_on_memory(self, memory, *, iter_id: int, ep_id: int):
        random.shuffle(memory)
        B = self.args["batch_size"]
        lam_H = self.args.get("entropy_bonus", 0.02)

        for i in range(0, len(memory), B):
            batch = memory[i : i + B]
            states, scalars, actions, returns = zip(*batch)

            dev = next(self.model.parameters()).device
            xx = torch.stack(states).to(dev)
            ss = torch.stack(scalars).to(dev)
            aa = torch.stack(actions).to(dev)
            vv = torch.stack(returns).to(dev)

            # ------- reward normalisation ----------------
            self._update_ret_stats(vv)
            vv_norm = (vv - self.ret_mean) / self._ret_std()

            # ------- forward / backward ------------------
            loss, pol, val, ent = self.model.fwd_train(xx, ss, aa, vv_norm, lambda_H=lam_H)

            self._log_batch_loss(pol, val, ent) 

            self.optimizer.zero_grad()
            loss.backward()

            # grad-clip DOPO backward
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                        max_norm=self.args.get("grad_clip", 1.0))

            self.optimizer.step()

        # scheduler una volta per epoca
        self.scheduler.step()
        self._flush_loss_meter(iter_id, ep_id)


    # ============================== loop =================================
    def learn(self):
        for it in range(self.args["num_iterations"]):
            # -------- self-play ----------------------------------------
            self.model.eval()
            for _ in trange(self.args["num_selfPlay_iterations"], desc=f"Iter {it} – self-play"):
                episode = self.self_play()          # lista di tuples
                self.replay.extend(episode)         # append nella FIFO

            # -------- training -----------------------------------------
            self.model.train()
            for ep in range(self.args["num_epochs"]):
                # campiona un minibatch random dal replay
                batch = random.sample(self.replay, k=min(self.args["batch_size"], len(self.replay)))
                self.train_on_memory(batch, iter_id=it, ep_id=ep + 1)
        
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), f"models/model_iter{it}.pt")
        # after all iterations – plot loss curves
        self._save_loss_plot("loss_curve.png")

    # ============================== plot =================================
    def _save_loss_plot(self, fname: str):
        if not self.curve:
            print("[warn] no loss data to plot – skipping figure")
            return

        labels, pol_l, val_l, tot_l = [], [], [], []
        for i, ep, pol, val, tot in self.curve:
            labels.append(f"{i}.{ep}")
            pol_l.append(pol)
            val_l.append(val)
            tot_l.append(tot)

        plt.figure(figsize=(9, 4))
        plt.plot(labels, pol_l, label="Policy CE")
        plt.plot(labels, val_l, label="Value MSE")
        plt.plot(labels, tot_l, label="Total", linewidth=2)
        plt.xlabel("Iter.Epoch")
        plt.ylabel("Loss")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"\n📈  curva loss salvata in →  {fname}\n")
