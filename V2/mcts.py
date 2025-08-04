# mcts.py
from __future__ import annotations
import math
import torch
from node import Node


class MCTS:
    def __init__(self, game, model, args):
        # inits
        self.game  = game
        self.model = model

        # Hyper
        self.c_puct        = args.get("C", 1.0)
        self.num_sim       = args.get("num_searches", 64)
        self.top_k         = args.get("top_k", None)
        
        # per debugging
        self.last_root: Node | None = None

    @torch.no_grad()
    def search(self, root_state):

        root = Node(state=root_state, parent=None, action_taken=None, prior=1.0)
        self.last_root = root

        self.expand(root, is_root=True)

        for _ in range(self.num_sim - 1):
            node = root

            while not node.is_leaf():
                node = node.select(self.c_puct)

            value, done = self.game.get_value_and_terminated(node.state, depth=node.depth, register=False)
            if done:
                node.backpropagate(value)
                continue

            self.expand(node, is_root=False)

        return self._select_action_from_root(root)

    
    @torch.no_grad()
    def expand(self, node: Node, is_root: bool = False) -> None:
        
        enc = self.game.get_encoded_state(node.state).unsqueeze(0)                                      
        scal = self.game.get_scalar().unsqueeze(0)                    

        # facciamo inferenza del modello
        action_idx, probs, value, top_idx, top_p = self.model.fwd_infer(enc, scal,legal_mask=None,top_k=self.top_k)

        priors = probs[0]
        actions = torch.arange(priors.size(-1), device=priors.device)
        
        # Se top_k non è None, abbiamo già top_idx/top_p pronti
        if self.top_k is not None:
            actions = top_idx[0]
            priors  = top_p[0]

        for gid, prior in zip(actions.tolist(), priors.tolist()):
            child_state = self.game.pensa_azione(node.state, gid)
            node.add_child(child_state, action=gid, prior=prior)

        # 5) Backpropagation del value stimato
        node.backpropagate(value.item())

    
    def _select_action_from_root(self, root: Node) -> int:
        visits = torch.tensor([c.visit_count for c in root.children], dtype=torch.float32)
        best = torch.argmax(visits).item()
        return root.children[best].action_taken
