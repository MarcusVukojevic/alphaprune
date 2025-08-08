# mcts.py
from __future__ import annotations
import torch
import math
from node import Node
from graphviz import Digraph
from IPython.display import display


class MCTS:
    def __init__(self, game, model, args):
        self.game  = game
        self.model = model

        # Hyper
        self.c_puct        = args.get("C", 1.0)
        self.num_sim       = args.get("num_searches", 25)
        self.top_k         = args.get("top_k", None)

        # Exploration & regularization
        self.root_noise_eps       = args.get("root_noise_eps", 0.25)
        self.root_dirichlet_alpha = args.get("root_dirichlet_alpha", 0.3)
        self.gamma                = args.get("gamma", 0.97)
        self.depth_penalty        = args.get("depth_penalty", 0.01)

        self.last_root: Node | None = None

    @torch.no_grad()
    def search(self, root_state):
        self._root_step = int(self.game.numero_mossa)
        root = Node(state=root_state, parent=None, action_taken=None, prior=1.0)
        self.last_root = root

        self.expand(root, is_root=True)
        for _ in range(self.num_sim - 1):
            node = root
            while not node.is_leaf():
                node = node.select(self.c_puct)

            abs_depth = self._root_step + node.depth
            value, done = self.game.get_value_and_terminated(
                node.state, depth=abs_depth, register=False
            )
            if done:
                node.backpropagate(value)
                continue

            self.expand(node, is_root=False)

        return self._select_action_from_root(root)

    def _scalar_for_node(self, node):
        steps_left = self.game.limite_mosse - (self._root_step + node.depth)
        return torch.tensor([steps_left], dtype=torch.float32, device=self.game.device)

    def reuse_subtree(self, taken_action: int):
        if not self.last_root:
            return False
        child = None
        for c in self.last_root.children:
            if c.action_taken == taken_action:
                child = c
                break
        if child is None:
            self.last_root = None
            return False

        child.parent = None
        base = child.depth
        stack = [child]
        while stack:
            n = stack.pop()
            n.depth -= base
            stack.extend(n.children)

        self.last_root = child
        self._root_step = int(self.game.numero_mossa)
        return True

    @torch.no_grad()
    def expand(self, node: Node, is_root: bool = False) -> None:
        enc  = self.game.get_encoded_state(node.state).unsqueeze(0)
        scal = self._scalar_for_node(node).unsqueeze(0)

        # LEGAL MASK: solo gate ancora a 1 (spegnibili), + anti-toggle parent/root
        parent_action = node.parent.action_taken if node.parent is not None else None
        lmask = self.game.legal_mask(node.state, is_root=is_root, parent_action=parent_action)

        action_idx, probs, value, top_idx, top_p = self.model.fwd_infer(
            enc, scal, legal_mask=lmask.unsqueeze(0), top_k=self.top_k
        )

        if self.top_k is not None:
            actions = top_idx[0].tolist()
            priors  = top_p[0].tolist()
        else:
            priors_t = probs[0]
            actions  = torch.arange(priors_t.size(-1), device=priors_t.device).tolist()
            priors   = priors_t.tolist()

        # Filtro ulteriore nel caso il modello ignorasse la mask
        ap = [(a, p) for a, p in zip(actions, priors) if bool(lmask[a].item())]
        if not ap:
            # se tutti illegali, niente figli -> backprop solo value
            node.backpropagate(value.item())
            return
        else:
            actions, priors = (list(t) for t in zip(*ap))

        # Dirichlet noise alla root
        #if is_root and self.root_noise_eps > 0 and len(priors) > 0:
        #    priors_t = torch.tensor(priors, device=enc.device, dtype=torch.float32)
        #    noise = torch.distributions.Dirichlet(
        #        torch.full_like(priors_t, self.root_dirichlet_alpha)
        #    ).sample()
        #    priors_t = (1 - self.root_noise_eps) * priors_t + self.root_noise_eps * noise
        #    priors = priors_t.tolist()

        # Shaping priors con PPL/Sparsity (soft)
        init_ppl  = self.game.initial_ppl
        target_s  = self.game.target_sparsity
        tol_frac  = float(getattr(self.game, "ppl_tol_frac", 0.10))
        alpha_ppl = float(getattr(self.game, "ppl_alpha", 0.05))
        w_s       = float(getattr(self, "w_sparsity", 0.3))
        s_beta    = float(getattr(self.game, "s_beta", 0.05))
        beta_pol  = float(getattr(self, "beta_policy", 1.0))
        beta_ppl  = float(getattr(self, "beta_ppl", 1.0))

        scored = []
        for gid, p in zip(actions, priors):
            child_state = self.game.pensa_azione(node.state, gid)
            # se illegal, skippa (in teoria già filtrato)
            if int(node.state[gid].item()) == 0:
                continue
            ppl_child = self.game.compute_ppl_for_state(child_state)
            rel = max(0.0, (ppl_child - init_ppl) / (init_ppl + 1e-8))
            ppl_w = 1.0 if rel <= tol_frac else math.exp(-(rel - tol_frac) / (alpha_ppl + 1e-12))

            if w_s > 0.0:
                s_child  = 1.0 - child_state.float().mean().item()
                s_peak   = math.exp(-abs(s_child - target_s) / (s_beta + 1e-12))
                progress = max(0.0, min(1.0, s_child / max(target_s, 1e-8)))
                s_term   = 0.7 * s_peak + 0.3 * progress
                combo    = (p ** beta_pol) * (ppl_w ** beta_ppl) * (1.0 + w_s * s_term)
            else:
                combo    = (p ** beta_pol) * (ppl_w ** beta_ppl)
            scored.append((gid, child_state, max(combo, 1e-12)))

        if not scored:
            node.backpropagate(value.item())
            return

        Z = sum(sc for _, _, sc in scored)
        invZ = 1.0 / (Z + 1e-12)

        for gid, child_state, sc in scored:
            node.add_child(child_state, action=gid, prior=float(sc * invZ))

        # backup value con sconto e costo per profondità
        q_est = (value.item() * (self.gamma ** node.depth)) - (self.depth_penalty * node.depth)
        node.backpropagate(q_est)

    def _select_action_from_root(self, root: Node) -> int:
        visits = torch.tensor([c.visit_count for c in root.children], dtype=torch.float32)
        best = torch.argmax(visits).item()
        return root.children[best].action_taken

    @torch.no_grad()
    def plot_search(self, root_state):
        self._root_step = int(self.game.numero_mossa)
        root = Node(state=root_state, parent=None, action_taken=None, prior=1.0)
        self.last_root = root
        self.expand(root, is_root=True)
        for _ in range(self.num_sim - 1):
            node = root
            while not node.is_leaf():
                node = node.select(self.c_puct)
            abs_depth = self._root_step + node.depth
            value, done = self.game.get_value_and_terminated(
                node.state, depth=abs_depth, register=False
            )
            if done:
                node.backpropagate(value)
                continue
            self.expand(node, is_root=False)

        best_action = self._select_action_from_root(root)

        dot = Digraph(format='png')
        dot.attr('node', shape='box', fontsize='10', fontname='Courier')

        def recurse(node):
            q = node.value_sum / node.visit_count if node.visit_count > 0 else 0.0
            lbl = (
                f"Azione: {node.action_taken}\\n"
                f"Visite: {node.visit_count}\\n"
                f"Q: {q:.2f}\\n"
                f"Prior: {node.prior:.2f}"
            )
            nid = str(id(node))
            dot.node(nid, lbl)
            for child in node.children:
                dot.edge(nid, str(id(child)))
                recurse(child)

        recurse(root)
        display(dot)
        return best_action
