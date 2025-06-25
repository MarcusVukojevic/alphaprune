# alpha_zero.py
import os, random, math, torch
from tqdm import trange, tqdm
from mcts import MCTS

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model     = model
        self.optimizer = optimizer
        self.game      = game
        self.args      = args
        self.mcts      = MCTS(game, args, model)

    # -------------------------------------------------------- #
    #   1.  SELF-PLAY → memoria (state, scalar, action, return)
    # -------------------------------------------------------- #
    @torch.no_grad()
    def self_play(self):
        memory = []
        state  = self.game.get_initial_state()

        while True:
            # MCTS → best azione
            action = self.mcts.search(state)              # tensor(2,)
            enc    = self.game.get_encoded_state(state)   # (T,N)
            scal   = self.game.get_scalar()               # (1,)
            memory.append((enc, scal, action))

            # aggiorna stato
            state = self.game.perform_action(action)
            reward, done = self.game.get_value_and_terminated(state)
            if done:
                final_reward = reward
                break

        # attacca il return a tutti i frame
        return [(st, sc, act, torch.tensor([final_reward])) for st, sc, act in memory]

    # -------------------------------------------------------- #
    #   2.  TRAIN sulla memoria
    # -------------------------------------------------------- #
    def train_on_memory(self, memory):
        random.shuffle(memory)
        B = self.args["batch_size"]

        for i in range(0, len(memory), B):
            batch = memory[i:i+B]
            states, scalars, actions, returns = zip(*batch)

            device = next(self.model.parameters()).device
            xx = torch.stack(states).to(device)
            ss = torch.stack(scalars).to(device)
            aa = torch.stack(actions).to(device)         
            vv = torch.stack(returns).to(device)         

            pol_loss, val_loss = self.model.fwd_train(xx, ss, aa, vv)
            loss = pol_loss + val_loss
            print("Loss: ", loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # -------------------------------------------------------- #
    #   3.  LOOP principale senza pre-train
    # -------------------------------------------------------- #
    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            self.model.eval()
            memory = []
            for _ in trange(self.args["num_selfPlay_iterations"], desc=f"Iter {iteration} – self-play"):
                memory += self.self_play()

            self.model.train()
            for ep in range(self.args["num_epochs"]):
                self.train_on_memory(memory)
                print(f"[Iter {iteration}] Epoch {ep+1}/{self.args['num_epochs']} – mem {len(memory)}")

            # opz: salva checkpoint
            os.makedirs("models", exist_ok=True)
            torch.save(self.model.state_dict(), f"models/model_iter{iteration}.pt")
