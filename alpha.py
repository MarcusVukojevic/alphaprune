import os
import random
import math
import torch
from tqdm import trange
import matplotlib.pyplot as plt

from mcts import MCTS


class AlphaZero:
    """Minimal AlphaZero-style trainer for PruneGame, now with
    – online reward normalisation (Welford)
    – running loss meter (policy / value / total)
    – automatic PNG plot of the loss curves at the end of training
    """
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
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

    def _log_batch_loss(self, pol_loss, val_loss):
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

    # ============================== self‑play =============================
    @torch.no_grad()
    def self_play(self):
        memory = []
        state = self.game.get_initial_state()

        while True:
            action = self.mcts.search(state)  # tensor([block, op])
            enc = self.game.get_encoded_state(state)  # (T, N)
            scal = self.game.get_scalar()  # (1,)
            memory.append((enc, scal, action))
            #memory.append((enc, scal, action, torch.tensor([self.game.reward]))) forse più informativo questo

            # step environment
            state = self.game.perform_action(action)
            reward, done = self.game.get_value_and_terminated(state)
            if done:
                final_reward = reward
                break

        # broadcast final reward to every frame
        return [(st, sc, act, torch.tensor([final_reward])) for st, sc, act in memory]

    # ============================== training ==============================
    def train_on_memory(self, memory, *, iter_id: int, ep_id: int):
        random.shuffle(memory)
        B = self.args["batch_size"]

        for i in range(0, len(memory), B):
            batch = memory[i : i + B]
            states, scalars, actions, returns = zip(*batch)

            dev = next(self.model.parameters()).device
            xx = torch.stack(states).to(dev)
            ss = torch.stack(scalars).to(dev)
            aa = torch.stack(actions).to(dev)
            vv = torch.stack(returns).to(dev)

            # -------- reward normalisation --------------------------------
            self._update_ret_stats(vv)
            vv_norm = (vv - self.ret_mean) / self._ret_std()

            # forward / backward -------------------------------------------
            pol_loss, val_loss = self.model.fwd_train(xx, ss, aa, vv_norm)
            loss = pol_loss + val_loss

            self._log_batch_loss(pol_loss, val_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # flush meter & store curve
        self._flush_loss_meter(iter_id, ep_id)

    # ============================== loop =================================
    def learn(self):
        for it in range(self.args["num_iterations"]):
            # ---------- collect fresh games ------------------------------
            self.model.eval()
            memory = []
            for _ in trange(self.args["num_selfPlay_iterations"], desc=f"Iter {it} – self‑play"):
                memory += self.self_play()

            # ---------- supervised / RL update --------------------------
            self.model.train()
            for ep in range(self.args["num_epochs"]):
                self.train_on_memory(memory, iter_id=it, ep_id=ep + 1)

            # ---------- optional: checkpoint ---------------------------
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
