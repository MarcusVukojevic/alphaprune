import os
import random
import math
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from collections import deque
from mcts import MCTS

class AlphaZero:
    def __init__(self, model, optimizer, game, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

        # ----------------- running statistics for reward ------------------
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.ret_count = 1e-4

        # ----------------- loss and metrics tracking ---------------------
        # Ora salviamo: (iter, epoch, pol_loss, val_loss, ent_loss, avg_reward_for_iter)
        self.history = []
        self._current_iter_reward = 0.0
        self._reset_loss_meter()

        self.replay = deque(maxlen=self.args.get("replay_size", 1000))

    def _update_ret_stats(self, returns: torch.Tensor):
        for r in returns.view(-1):
            self.ret_count += 1.0
            delta = r.item() - self.ret_mean
            self.ret_mean += delta / self.ret_count
            self.ret_var += delta * (r.item() - self.ret_mean)

    def _ret_std(self):
        return math.sqrt(max(self.ret_var / (self.ret_count - 1), 1e-6))

    def _reset_loss_meter(self):
        """Resetta gli accumulatori per le loss di un'epoca."""
        self._loss_pol, self._loss_val, self._loss_ent = [], [], []

    def _log_batch_loss(self, pol_loss, val_loss, entr_loss):
        """Aggiunge le loss di un batch agli accumulatori."""
        self._loss_pol.append(pol_loss.item())
        self._loss_val.append(val_loss.item())
        self._loss_ent.append(entr_loss.item())

    def _flush_loss_meter(self, iter_id: int, ep_id: int):
        """
        Calcola le medie delle loss, le stampa e le salva nella history.
        """
        if not self._loss_pol:
            return
            
        m_pol = sum(self._loss_pol) / len(self._loss_pol)
        m_val = sum(self._loss_val) / len(self._loss_val)
        m_ent = sum(self._loss_ent) / len(self._loss_ent)
        
        # Stampa a console le metriche aggiornate
        print(f"[Iter {iter_id}  Ep {ep_id}]  pol={m_pol:.4f}  val={m_val:.4f}  ent={m_ent:.4f}")
        
        # Salva tutti i dati nella history
        self.history.append((iter_id, ep_id, m_pol, m_val, m_ent, self._current_iter_reward))
        self._reset_loss_meter()

    @torch.no_grad()
    def self_play(self):

        trajectory = []
        state = self.game.get_initial_state()

        # Esegue la partita fino alla fine
        while True:
            action = self.mcts.search(state)
            
            # Salva stato e policy MCTS prima di fare la mossa
            enc = self.game.get_encoded_state(state)
            scal = self.game.get_scalar()
            
            visits = torch.zeros(self.model.num_blocks * self.model.num_ops, device=enc.device)
            for child in self.mcts.last_root.children:
                b, o = child.action_taken
                visits[b * self.model.num_ops + o] = child.visit_count
            pi = visits / visits.sum()
            

            # Esegui la mossa
            state = self.game.perform_action(action)
            print("azione: ", action)
            # La variabile 'reward' qui è l'accumulo, ma non la useremo come target finale
            reward, done = self.game.get_value_and_terminated(state)
            
            #trajectory.append((enc.cpu(), scal.cpu(), pi.cpu(), torch.tensor(reward, dtype=torch.float32)))
            
            trajectory.append((enc.cpu(), scal.cpu(), pi.cpu()))
            if done:
                break
        
        # --- INIZIO MODIFICA ---
        # Alla fine della partita, calcola il valore oggettivo dello stato finale.
        # Questo valore è un target di training più stabile e pulito.
        final_sparsity = 1.0 - state.float().mean().item()
        final_kl_div = self.game.kl_div
        final_phi = final_sparsity - self.game.beta * final_kl_div
        
        # Questo è il nuovo "return" (z) che la value network imparerà a predire
        final_return_value = torch.tensor([final_phi])
        # --- FINE MODIFICA ---
        
        # Propaga questo valore finale a tutti gli stati della traiettoria
        #return [(st, sc, pi, rew) for st, sc, pi, rew in trajectory]
        return [(st, sc, pi, final_return_value) for st, sc, pi in trajectory]

    def train_on_memory(self, memory, *, iter_id: int, ep_id: int):
        """Esegue un'epoca di training campionando dalla memoria di replay."""
        random.shuffle(memory)
        B = self.args["batch_size"]
        lam_H = self.args.get("entropy_bonus", 0.02)

        for i in range(0, len(memory), B):
            batch = memory[i:(i + B)]
            states, scalars, actions, returns = zip(*batch)

            dev = next(self.model.parameters()).device
            xx, ss, aa, vv = (
                torch.stack(states).to(dev),
                torch.stack(scalars).to(dev),
                torch.stack(actions).to(dev),
                torch.stack(returns).to(dev),
            )

            self._update_ret_stats(vv)
            vv_norm = (vv - self.ret_mean) / self._ret_std()

            loss, pol, val, ent = self.model.fwd_train(xx, ss, aa, vv_norm, lambda_H=lam_H)
            self._log_batch_loss(pol, val, ent)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.get("grad_clip", 1.0))
            self.optimizer.step()

        self.scheduler.step()
        self._flush_loss_meter(iter_id, ep_id)

    def learn(self):
        """Ciclo di apprendimento principale."""
        for it in range(self.args["num_iterations"]):
            # -------- Self-Play ----------------------------------------
            self.model.eval()
            episode_rewards = []
            print(f"\n--- Iteration {it}: Self-Play ---")
            for _ in trange(self.args["num_selfPlay_iterations"], desc=f"Iter {it} – self-play"):
                episode = self.self_play()
                self.replay.extend(episode)
                episode_rewards.append(episode[-1][3].item()) # Salva il reward finale

            if episode_rewards:
                self._current_iter_reward = sum(episode_rewards) / len(episode_rewards)

            # -------- Training -----------------------------------------
            self.model.train()
            print(f"--- Iteration {it}: Training (Avg Reward: {self._current_iter_reward:.4f}) ---")
            for ep in range(self.args["num_epochs"]):
                batch = random.sample(self.replay, k=min(self.args["batch_size"], len(self.replay)))
                self.train_on_memory(batch, iter_id=it, ep_id=ep + 1)
        
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), f"models/model_iter{it}.pt")
        self._save_loss_plot("loss_curve.png")
        self.mcts.render_mcts_tree(self.mcts.last_root, filename="mcts_iter0", max_depth=4)

    def _save_loss_plot(self, fname: str):
        """Salva il grafico finale con loss e reward."""
        if not self.history:
            print("[warn] no history data to plot – skipping figure")
            return

        labels, pol_l, val_l, ent_l, reward_l = [], [], [], [], []
        for i, ep, pol, val, ent, rew in self.history:
            labels.append(f"{i}.{ep}")
            pol_l.append(pol)
            val_l.append(val)
            ent_l.append(ent)
            reward_l.append(rew)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Asse Y sinistro per le Loss
        ax1.set_xlabel("Iter.Epoch")
        ax1.set_ylabel("Loss", color="tab:blue")
        ax1.plot(labels, pol_l, label="Policy Loss", color="tab:blue", linestyle='-')
        ax1.plot(labels, val_l, label="Value Loss", color="tab:cyan", linestyle=':')
        ax1.plot(labels, ent_l, label="Entropy Loss", color="tab:gray", linestyle='--')
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        # Asse Y destro per il Reward
        ax2 = ax1.twinx()
        ax2.set_ylabel("Average Reward", color="tab:red")
        ax2.plot(labels, reward_l, label="Avg Episode Reward", color="tab:red", marker='.', linestyle='None')
        ax2.tick_params(axis='y', labelcolor="tab:red")

        # Legenda unificata
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        fig.tight_layout()
        plt.title("Training History: Losses and Rewards")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"\n📈 Curva di training salvata in → {fname}\n")