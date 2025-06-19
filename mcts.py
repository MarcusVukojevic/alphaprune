# mcts.py  – versione aggiornata
import torch
from node import Node

class MCTS:
    def __init__(self, game, args, model):
        self.game   = game
        self.args   = args
        self.model  = model
        self.Cpuct  = args.get("C", 1.0)
        self.K      = args.get("top_k", 64)

    # ------------------------------------------------------------
    @torch.no_grad()
    def search(self, root_state):
        root = Node(state=root_state, parent=None, action_taken=None, prior=1.0)
        self._expand(root)                       # primo livello

        for _ in range(self.args["num_searches"]):
            node = root
            # 1) SELEZIONE
            while node.children:
                node = node.select(self.Cpuct)

            # 2) VALORE & TERMINAZIONE
            value, done = self.game.get_value_and_terminated(node.state)

            # 3) ESPANSIONE
            if not done:
                self._expand(node)
                value, _ = self.game.get_value_and_terminated(node.state)

            # 4) BACK-PROP
            node.backpropagate(value)

        # azione con più visite
        best_child = max(root.children, key=lambda c: c.visit_count)
        return best_child.action_taken

    # ------------------------------------------------------------
    def _expand(self, node):
        # ----- prepara input per la rete ------ #
        enc = self.game.get_encoded_state(node.state).unsqueeze(0)   # (1,T,N)
        scl = self.game.get_scalar().unsqueeze(0)                    # (1,1)

        dev = next(self.model.parameters()).device   # << nuovo modo
        enc, scl = enc.to(dev), scl.to(dev)

        # ----- inference ------ #
        acts, priors, _ = self.model.fwd_infer(enc, scl, top_k=self.K)
        acts, priors = acts[0], priors[0]     # (K,2) , (K,)

        # ----- genera figli ------ #
        for a, p in zip(acts, priors):
            child_state = self.game.get_next_state(node.state.clone(), a)
            node.add_child(child_state, action=a.clone(), prior=p.item())
