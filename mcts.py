# mcts.py  – versione aggiornata
import torch
from node import Node


TOGGLE = 0
SKIP_BLOCK = 1
PASS   = 2

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

        self.last_root = root = Node(state=root_state, parent=None, action_taken=None, prior=1.0)
        root_v = self._expand(root)              # primo livello: stima V(s₀)

        for _ in range(self.args["num_searches"]):
            node = root
            # 1) SELEZIONE
            while node.children:
                node = node.select(self.Cpuct)

            # 2) VALORE & TERMINAZIONE
            reward, done = self.game.get_value_and_terminated(node.state)

            ## 3) ESPANSIONE
            #if not done:
            #    self._expand(node)
            #    reward, _ = self.game.get_value_and_terminated(node.state)

            if not done:
                leaf_v = self._expand(node)
                reward = leaf_v                       # valore stimato

            # 4) BACK-PROP
            node.backpropagate(reward)

        # azione con più visite
        best_child = max(root.children, key=lambda c: c.visit_count)
        #print(f"[root] visits:", {tuple(c.action_taken.tolist()): c.visit_count for c in root.children[:6]})
        #print(f"[root] Q:", {tuple(c.action_taken.tolist()): round(c.get_q(),3) for c in root.children[:6]})
        return best_child.action_taken

    def _expand(self, node):
        # ----- prepara input per la rete ------ #
        enc = self.game.get_encoded_state(node.state).unsqueeze(0)   # (1,T,N)
        scl = self.game.get_scalar().unsqueeze(0)                    # (1,1)

        dev = next(self.model.parameters()).device
        enc, scl = enc.to(dev), scl.to(dev)


        # ----- inference ------ #
        #acts, priors, _ = self.model.fwd_infer(enc, scl, top_k=self.K)
        # patch
        acts, priors, leaf_v = self.model.fwd_infer(enc, scl, top_k=self.K)
        acts, priors = acts[0], priors[0]         # (K,2) , (K,)

        #  filtro PASS nelle prime 3 mosse 
        if self.game.time_stamp < 3:
            mask = acts[:, 1] != PASS             # op == 3  → PASS
            if mask.any():                        # se rimane qualcosa
                acts, priors = acts[mask], priors[mask]
            else:                                 # erano tutti PASS → tieni il primo
                acts, priors = acts[:1], priors[:1]

        #  Dirichlet noise sulla radice per migliorare la scelta delle azioni
        if node.parent is None:
            eps   = self.args.get("root_dir_eps", 0.3)
            alpha = self.args.get("root_dir_alpha", 0.3)
            
            if priors.device.type == "mps":
                conc  = torch.full((priors.numel(),), alpha, dtype=priors.dtype, device="cpu")
                noise = torch.distributions.Dirichlet(conc).sample().to(priors.device)
            else:                                          
                noise = torch.distributions.Dirichlet(torch.full_like(priors, alpha)).sample()
            priors = priors * (1 - eps) + noise * eps

        # ----- genera figli ------ #
        for a, p in zip(acts, priors):
            child_state = self.game.get_next_state(node.state.clone(), a)
            node.add_child(child_state, action=a.clone(), prior=p.item())

        return leaf_v.item()
