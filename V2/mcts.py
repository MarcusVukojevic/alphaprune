# mcts.py
from __future__ import annotations
import torch
from node import Node
from graphviz import Digraph
from IPython.display import display
import math

class MCTS:
    def __init__(self, game, model, args):
        # inits
        self.game  = game
        self.model = model

        # Hyper
        self.c_puct        = args.get("C", 1.0)
        self.num_sim       = args.get("num_searches", 64)
        self.top_k         = args.get("top_k", None)

        # Exploration & regularization
        self.root_noise_eps       = args.get("root_noise_eps", 0.25)
        self.root_dirichlet_alpha = args.get("root_dirichlet_alpha", 0.3)
        self.gamma                = args.get("gamma", 1.0)           # discount on model value
        self.depth_penalty        = args.get("depth_penalty", 1)   # small per-step cost

        # per debugging
        self.last_root: Node | None = None

    @torch.no_grad()
    def search(self, root_state):
        # contabilizza i passi reali già effettuati
        self._root_step = int(self.game.numero_mossa)

        root = Node(state=root_state, parent=None, action_taken=None, prior=1.0)
        self.last_root = root

        self.expand(root, is_root=True)

        for _ in range(self.num_sim - 1):
            node = root

            while not node.is_leaf():
                node = node.select(self.c_puct)

            # usa passi ASSOLUTI (root_step + depth) per il terminal check
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
        # passi rimanenti = limite - (root_step + depth_nodo)
        steps_left = self.game.limite_mosse - (self._root_step + node.depth)
        return torch.tensor([steps_left], dtype=torch.float32, device=self.game.device)

    @torch.no_grad()
    def expand_prima(self, node: Node, is_root: bool = False) -> None:
        enc = self.game.get_encoded_state(node.state).unsqueeze(0)
        scal = self.game.get_scalar().unsqueeze(0)

        action_idx, probs, value, top_idx, top_p = self.model.fwd_infer(
            enc, scal, legal_mask=None, top_k=self.top_k
        )

        priors = probs[0]
        actions = torch.arange(priors.size(-1), device=priors.device)

        if self.top_k is not None:
            actions = top_idx[0]
            priors  = top_p[0]

        for gid, prior in zip(actions.tolist(), priors.tolist()):
            child_state = self.game.pensa_azione(node.state, gid)
            node.add_child(child_state, action=gid, prior=prior)

        # Backpropagation del value stimato
        node.backpropagate(value.item())

    def reuse_subtree(self, taken_action: int):
        """Promuovi a nuova radice il figlio corrispondente all'azione scelta
        e ricalibra le profondità del sottoalbero."""
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

        # stacca dalla vecchia radice
        child.parent = None

        # rebase depth: porta il sottoalbero a partire da 0
        base = child.depth
        stack = [child]
        while stack:
            n = stack.pop()
            n.depth -= base
            stack.extend(n.children)

        self.last_root = child
        # aggiorna lo step reale alla nuova radice
        self._root_step = int(self.game.numero_mossa)
        return True

    @torch.no_grad()
    def expand(self, node: Node, is_root: bool = False) -> None:
        enc  = self.game.get_encoded_state(node.state).unsqueeze(0)
        scal = self._scalar_for_node(node).unsqueeze(0)

        # ---- LEGAL MASK anti-toggle ----
        # Maschera il revert immediato rispetto al padre nel tree,
        # e alla radice evita di tornare subito all'ultima azione reale.
        lmask = torch.ones_like(node.state, dtype=torch.bool)
        parent_action = node.parent.action_taken if node.parent is not None else None
        if parent_action is not None and 0 <= parent_action < lmask.numel():
            lmask[parent_action] = False
        # evita flip-flop tra mosse reali consecutive se il gioco espone last_real_action
        last_real = getattr(self.game, "last_real_action", None)
        if is_root and last_real is not None and 0 <= last_real < lmask.numel():
            lmask[last_real] = False

        # inferenza del modello (se non supporta la mask internamente, filtriamo dopo)
        action_idx, probs, value, top_idx, top_p = self.model.fwd_infer(enc, scal, legal_mask=lmask.unsqueeze(0), top_k=self.top_k)

        # azioni + priors dalla policy del modello
        if self.top_k is not None:
            actions = top_idx[0].tolist()
            priors  = top_p[0].tolist()
        else:
            priors_t = probs[0]
            actions  = torch.arange(priors_t.size(-1), device=priors_t.device).tolist()
            priors   = priors_t.tolist()

        # filtro lato MCTS nel caso il modello ignori la mask
        ap = [(a, p) for a, p in zip(actions, priors) if bool(lmask[a].item())]
        if not ap:
            # fallback: abilita tutto tranne il parent_action (alla radice non c'è parent)
            N = node.state.numel()
            fallback_actions = [i for i in range(N) if i != (parent_action if parent_action is not None else -1)]
            if not fallback_actions:  # degenerato (N=1)
                fallback_actions = list(range(N))
            uniform = 1.0 / max(len(fallback_actions), 1)
            actions, priors = fallback_actions, [uniform] * len(fallback_actions)
        else:
            actions, priors = (list(t) for t in zip(*ap))


        # ---- Scoring custom (policy × ppl × sparsity) ----
        init_ppl  = self.game.initial_ppl
        target_s  = self.game.target_sparsity
        tol_frac  = float(getattr(self, "ppl_tol_frac", 0.02))
        alpha_ppl = float(getattr(self, "ppl_alpha", 0.02))   
        w_s       = float(getattr(self, "w_sparsity", 0.3))
        s_beta    = float(getattr(self, "s_beta", 0.05))
        beta_pol  = float(getattr(self, "beta_policy", 1.0))
        beta_ppl  = float(getattr(self, "beta_ppl", 1.0))

        scored = []
        for gid, p in zip(actions, priors):
            child_state = self.game.pensa_azione(node.state, gid)
            ppl_child = self.game.compute_ppl_for_state(child_state)
            rel = max(0.0, (ppl_child - init_ppl) / (init_ppl + 1e-8))

            # Plateau: entro tol_frac non penalizzi, fuori esponenziale sul surplus
            ppl_w = 1.0 if rel <= tol_frac else math.exp(-(rel - tol_frac) / (alpha_ppl + 1e-12))

            # Sparsity shaping (peak+progress), gated dalla PPL
            if w_s > 0.0:
                s_child  = 1.0 - child_state.float().mean().item()
                s_peak   = math.exp(-abs(s_child - target_s) / (s_beta + 1e-12))
                progress = max(0.0, min(1.0, s_child / max(target_s, 1e-8)))
                s_term   = 0.7 * s_peak + 0.3 * progress
                combo    = (p ** beta_pol) * (ppl_w ** beta_ppl) * (1.0 + w_s * s_term)
            else:
                combo    = (p ** beta_pol) * (ppl_w ** beta_ppl)

            scored.append((gid, child_state, max(combo, 1e-12)))

        # normalizza e aggiungi figli, con guardia su Z≈0
        Z = sum(sc for _, _, sc in scored)
        if Z <= 1e-12:
            # fallback uniforme tra le azioni generate
            scored = [(gid, st, 1.0) for gid, st, _ in scored]
            Z = float(len(scored))
        invZ = 1.0 / (Z + 1e-12)

        for gid, child_state, sc in scored:
            node.add_child(child_state, action=gid, prior=float(sc * invZ))

        # backup del value del modello (scontato e con costo per profondità)
        q_est = (value.item() * (self.gamma ** node.depth)) - (self.depth_penalty * node.depth)
        node.backpropagate(q_est)

    def _select_action_from_root(self, root: Node) -> int:
        visits = torch.tensor([c.visit_count for c in root.children], dtype=torch.float32)
        best = torch.argmax(visits).item()
        return root.children[best].action_taken

    @torch.no_grad()
    def plot_search(self, root_state):
        # contabilizza i passi reali già effettuati
        self._root_step = int(self.game.numero_mossa)

        # 1) Creazione e ricerca
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

        # 2) Selezione dell’azione d’uscita
        best_action = self._select_action_from_root(root)

        # 3) Costruzione del grafo
        dot = Digraph(format='png')
        dot.attr('node', shape='box', fontsize='10', fontname='Courier')

        def recurse(node):
            # etichetta compatta
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

        # 4) Visualizzazione inline in Jupyter
        display(dot)

        return best_action
