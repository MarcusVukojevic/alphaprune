# mcts.py
from __future__ import annotations
import math, random
import torch
from node import Node
from graphviz import Digraph
#from IPython.display import display

class MCTS:
    def __init__(self, game, model, args):
        self.game  = game
        self.model = model

        self.c_puct        = args.get("C", 1.0)
        self.num_sim       = args.get("num_searches", 25)
        self.top_k         = args.get("top_k", None)

        self.root_noise_eps       = args.get("root_noise_eps", 0.25)
        self.root_dirichlet_alpha = args.get("root_dirichlet_alpha", 0.3)
        self.gamma                = args.get("gamma", 0.97)
        self.depth_penalty        = args.get("depth_penalty", 0.01)
        self.root_select_temp     = args.get("root_select_temp", 1.0)

        self.w_sparsity           = args.get("w_sparsity", 0.3)
        self.beta_policy          = args.get("beta_policy", 1.0)
        self.beta_ppl             = args.get("beta_ppl", 1.0)  # usato anche per KL/MSE come "qualità"

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

        ap = [(a, p) for a, p in zip(actions, priors) if bool(lmask[a].item())]
        if not ap:
            node.backpropagate(value.item())
            return
        else:
            actions, priors = (list(t) for t in zip(*ap))

        # --- Dirichlet root noise: CPU-safe, poi torna su device ---
        if is_root and self.root_noise_eps > 0 and len(priors) > 0:
            pri_cpu = torch.tensor(priors, dtype=torch.float32, device="cpu")
            alpha   = torch.full_like(pri_cpu, float(self.root_dirichlet_alpha), device="cpu")
            noise   = torch.distributions.Dirichlet(alpha).sample()   # CPU
            pri_cpu = (1 - self.root_noise_eps) * pri_cpu + self.root_noise_eps * noise
            priors_t = pri_cpu.to(enc.device)
        else:
            priors_t = torch.tensor(priors, dtype=torch.float32, device=enc.device)

        # --- Jitter anti-tie e normalizza ---
        priors_t = priors_t + 1e-3 * torch.rand_like(priors_t)
        priors_t = priors_t / priors_t.sum().clamp_min(1e-12)
        priors = priors_t.tolist()

        # --- Randomizza l'ordine dei figli per evitare bias sugli indici ---
        if len(priors) > 1:
            perm = torch.randperm(len(priors), device=enc.device)
            actions = [actions[i] for i in perm.tolist()]
            priors  = [priors[i]  for i in perm.tolist()]

        # ---- Scoring custom (policy × qualità × sparsity) ----
        target_s  = self.game.target_sparsity
        tol_frac  = float(getattr(self.game, "ppl_tol_frac", 0.10))  # usato solo se eval_mode="ppl"
        alpha_map = {
            "ppl": getattr(self.game, "ppl_alpha", 0.05),
            "kl" : getattr(self.game, "kl_alpha", 0.02),
            "mse": getattr(self.game, "mse_alpha", 0.02),
        }
        alpha_val = alpha_map.get(self.game.eval_mode, 0.05)
        s_beta    = float(getattr(self.game, "s_beta", 0.05))
        w_s       = float(getattr(self, "w_sparsity", 0.3))
        beta_pol  = float(getattr(self, "beta_policy", 1.0))
        beta_q    = float(getattr(self, "beta_ppl", 1.0))  # riuso come peso qualità

        scored = []
        for gid, p in zip(actions, priors):
            child_state = self.game.pensa_azione(node.state, gid)
            if int(node.state[gid].item()) == 0:
                continue

            # qualità: ppl/kl/mse più bassa è meglio → peso exp(-x/alpha) o plateau ppl
            m_child = self.game.metric_from_state_or_cache(child_state)
            if self.game.eval_mode == "ppl":
                init_ppl = getattr(self.game, "initial_ppl")
                rel = max(0.0, (m_child - init_ppl) / (init_ppl + 1e-8))
                q_w = 1.0 if rel <= tol_frac else math.exp(-(rel - tol_frac) / (alpha_val + 1e-12))
            else:
                q_w = math.exp(- m_child / (alpha_val + 1e-12))

            if w_s > 0.0:
                s_child  = 1.0 - child_state.float().mean().item()
                s_peak   = math.exp(-abs(s_child - target_s) / (s_beta + 1e-12))
                progress = max(0.0, min(1.0, s_child / max(target_s, 1e-8)))
                s_term   = 0.7 * s_peak + 0.3 * progress
                combo    = (p ** beta_pol) * (q_w ** beta_q) * (1.0 + w_s * s_term)
            else:
                combo    = (p ** beta_pol) * (q_w ** beta_q)

            scored.append((gid, child_state, max(combo, 1e-12)))

        if not scored:
            node.backpropagate(value.item())
            return

        Z = sum(sc for _, _, sc in scored)
        invZ = 1.0 / (Z + 1e-12)

        for gid, child_state, sc in scored:
            node.add_child(child_state, action=gid, prior=float(sc * invZ))

        q_est = (value.item() * (self.gamma ** node.depth)) - (self.depth_penalty * node.depth)
        node.backpropagate(q_est)

    def _select_action_from_root(self, root: Node) -> int:
        visits = torch.tensor([c.visit_count for c in root.children], dtype=torch.float32)
        tau = getattr(self, "root_select_temp", 1.0)
        if tau and tau > 0:
            probs = torch.softmax(visits / tau, dim=0)
            idx = torch.multinomial(probs, num_samples=1).item()
        else:
            idx = torch.argmax(visits).item()
        return root.children[idx].action_taken

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
        #display(dot)
        return best_action
