# prune_game.py
import math
from collections import deque, OrderedDict

import torch
import torch.nn as nn

from patches import _patch_gpt2_block, _patch_llama_block
from utils import load_model, build_calib_dataset


class PruneGame:
    # funzione per applicare la patch al modello
    @staticmethod
    def _ensure_patched(model: nn.Module):
        for blk in model.modules():
            if hasattr(blk, "attn") and hasattr(blk, "mlp") and hasattr(blk, "ln_1"):
                _patch_gpt2_block(blk)
            elif hasattr(blk, "self_attn") and hasattr(blk, "mlp") and hasattr(blk, "input_layernorm"):
                _patch_llama_block(blk)
        return model

    # funzione che restituisce stato e lista di ResidualGates
    @staticmethod
    def _collect_gates(model):
        ptrs = []
        for blk in model.modules():
            if hasattr(blk, "g_mha") and hasattr(blk, "g_ffn"):
                ptrs.extend([blk.g_mha, blk.g_ffn])  # due per layer
        if not ptrs:
            raise ValueError("Patch failed – no gates found.")
        state = torch.ones(len(ptrs), dtype=torch.int8)
        return state, ptrs

    def __init__(self, args):
        # Var iniziali/utili
        self.model_name = args["model_name"]
        self.device = args["device"]
        self.eight_bit = args.get("eight_bit", False)
        self.target_sparsity = args["target_sparsity"]
        self.numero_mossa = 0
        self.limite_mosse = args["n_mosse_massimo"]
        self.nome_dataset = args["name_dataset"]
        self.eval_mode = args["eval_mode"]
        self.args = args

        # Tolleranze/parametri consistenti
        self.ppl_tol_frac = args.get("ppl_tol_frac", 0.10)  # non usata nel reward fail (vedi sotto)
        self.ppl_alpha    = args.get("ppl_alpha", 0.05)
        self.s_beta       = args.get("s_beta", 0.05)
        self.eps_sparsity_bonus = args.get("eps_sparsity_bonus", 0.2)

        # modello e tokenizer
        self.model_victim = PruneGame._ensure_patched(
            load_model(self.model_name, device=self.device, eightbit=self.eight_bit)
        )
        if hasattr(self.model_victim.config, "use_cache"):
            self.model_victim.config.use_cache = False
        self.tokenizer = self.model_victim.tokenizer

        # stato iniziale (tutti i gate ON)
        self.state, self.gates = PruneGame._collect_gates(self.model_victim)
        self.state = self.state.to(self.device)
        self.initial_state = self.state.clone()

        # dataset di calibrazione
        self.dataset = build_calib_dataset(self.nome_dataset, self.tokenizer,
                                           split="validation", nsamples=5, seq_len=128)

        # metriche iniziali
        self.ppl = self.compute_ppl()
        self.initial_ppl = self.ppl
        print("--> ppl iniziale:", self.ppl)

        if self.eval_mode == "mse":
            self._cache_teacher_full()
            self.scale_mse = (self.teacher_full ** 2).mean().item()
        self.mse = 0

        # per debug e logging
        self.history = deque(maxlen=self.limite_mosse - 1)
        self.history.appendleft(self.initial_state)
        self.reward_history = []
        self.ppl_history = []
        self.mse_history = []
        self.last_real_action = None

    def _cache_teacher_full(self, batch_size=64, dtype=torch.float16):
        self.model_victim.eval()
        ref_lp, inputs = [], []
        with torch.no_grad():
            for i in range(0, len(self.dataset), batch_size):
                j   = min(i + batch_size, len(self.dataset))
                inp = torch.stack(self.dataset[i:j], dim=0).long().to(self.device)
                lp  = torch.log_softmax(self.model_victim(inp).logits, dim=-1)  # (B,L,V)
                ref_lp.append(lp.to(dtype).cpu())
                inputs.append(inp.cpu())
        self.teacher_full = torch.cat(ref_lp, dim=0)      # (N,L,V) fp16 CPU
        self.calib_inputs_cpu = torch.cat(inputs, dim=0)

    @torch.no_grad()
    def compute_mse(self, batch_size=64) -> float:
        self.model_victim.eval()
        tot, count = 0.0, 0
        N = self.teacher_full.size(0)
        for i in range(0, N, batch_size):
            j   = min(i + batch_size, N)
            inp = self.calib_inputs_cpu[i:j].to(self.device)
            lp  = torch.log_softmax(self.model_victim(inp).logits, dim=-1)
            ref = self.teacher_full[i:j].to(self.device, dtype=lp.dtype)
            mse = torch.mean((lp - ref) ** 2).item()
            bsz = (j - i)
            tot += mse * bsz
            count += bsz
        return tot / max(count, 1)

    def reset_game(self):
        # reset stato e gates
        self.state.copy_(self.initial_state)
        for g in self.gates:
            g.alpha.data.fill_(1.0)

        self.ppl = self.initial_ppl
        self.initial_mse = 0
        self.numero_mossa = 0
        self.history = deque(maxlen=self.limite_mosse - 1)
        self.history.appendleft(self.initial_state)
        return self.state

    def ppl_from_state_or_cache(self, state):
        if state.data_ptr() == self.state.data_ptr():  # stesso tensore
            return float(self.ppl)
        return self.compute_ppl_for_state(state)

    @torch.no_grad()
    def get_value_and_terminated(self, state, depth=None, register=False):
        """
        Episodio termina:
          - subito quando raggiungo la sparsity target (early stop, reward 0.7..1.0 in base alla PPL)
          - a limite mosse: SE NON ho raggiunto la sparsity → reward basso (progresso * PPL), mai vicino a 1
        """
        steps = self.numero_mossa if depth is None else depth

        # metriche sullo stato ipotetico
        s = 1.0 - state.float().mean().item()
        target = self.target_sparsity
        ppl_now = self.ppl_from_state_or_cache(state)

        # PPL: distanza relativa
        rel   = max(0.0, (ppl_now - self.initial_ppl) / (self.initial_ppl + 1e-8))
        ppl_term = math.exp(- rel / (self.ppl_alpha + 1e-12))  # (0,1]

        hit_s = (s >= target)
        limit = (steps >= self.limite_mosse)

        # EARLY STOP: appena raggiungi la sparsity, chiudi episodio
        if hit_s:
            # premia PPL migliore: 0.7–1.0
            reward = min(1.0, 0.7 + 0.3 * ppl_term)
            if register:
                self.reward_history.append(float(reward))
            return float(reward), True

        if not limit:
            return 0.0, False

        # Limite mosse senza sparsity: reward SEVERO (niente “premio PPL=1”)
        # Progresso normalizzato verso il target (0..1), poi compresso
        progress = max(0.0, min(1.0, s / max(target, 1e-8)))
        # Bilancia con PPL ma MAI alto; 0..0.4 circa
        reward = 0.4 * (progress * (0.5 + 0.5 * ppl_term))
        if register:
            self.reward_history.append(float(reward))
        return float(reward), True

    # funzione che dato uno stato restituisce il tensore da passare al modello
    def get_encoded_state(self, state: torch.Tensor):
        T, N = self.limite_mosse, state.numel()
        enc  = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        enc[0] = state.float()  # stato corrente
        for t, past_state in enumerate(self.history, start=1):
            if t >= T:
                break
            enc[t] = past_state.float()
        return enc

    def get_scalar(self):
        return torch.tensor([self.limite_mosse - self.numero_mossa],
                            dtype=torch.float32, device=self.device)

    # --- AZIONE MONOTONA: solo spegnere (1->0). Se già 0 è illegale. ---
    def disable_gate(self, gid: int):
        if int(self.state[gid].item()) == 0:
            return False  # azione illegale
        gate = self.gates[gid]
        gate.alpha.data.fill_(0.0)
        self.state[gid] = 0
        return True

    def do_action(self, gid: int):
        ok = self.disable_gate(gid)
        if not ok:
            # se il planner propone un'azione illegale, non cambiamo stato ma consumiamo lo step
            # (puoi anche scegliere di non consumare lo step; qui consumiamo per punire le scelte inutili)
            self.numero_mossa += 1
            self.last_real_action = gid
            self.history.appendleft(self.state.clone())
        else:
            self.numero_mossa += 1
            self.last_real_action = gid
            self.history.appendleft(self.state.clone())

        if self.eval_mode == "ppl":
            self.ppl = self.compute_ppl()
            self.ppl_history.append(self.ppl)
        elif self.eval_mode == "mse":
            self.mse = self.compute_mse()
            self.mse_history.append(self.mse)

        return self.state

    def legal_mask(self, state, is_root: bool, parent_action: int | None):
        """
        Azioni legali = gate ancora a 1 (spegnibili).
        Evita anche il revert immediato rispetto al padre/ultima reale (per robustezza,
        ma con monotonia è ridondante).
        """
        mask = state.bool().clone()  # True solo dove state==1
        if is_root and self.last_real_action is not None:
            mask[self.last_real_action] = False
        if parent_action is not None:
            mask[parent_action] = False
        return mask

    # sarebbe "spegni gate" su uno stato ipotetico (non muta self.state)
    def pensa_azione(self, state: torch.Tensor, action: int):
        if int(state[action].item()) == 0:
            return state  # illegal -> ritorna invariato (MCTS filtrerà via)
        nuovo_stato = state.clone()
        nuovo_stato[action] = 0
        return nuovo_stato

    @torch.no_grad()
    def compute_ppl(self, batch_size: int = 64) -> float:
        self.model_victim.eval()
        total_nll, total_tok = 0.0, 0
        loader = self.dataset
        dev = self.device
        for i in range(0, len(loader), batch_size):
            j = min(i + batch_size, len(loader))
            inp = torch.stack(loader[i:j], dim=0).long().to(dev)
            outputs = self.model_victim(inp, labels=inp, use_cache=False)
            loss = outputs.loss.float()           # già shiftata
            total_nll += loss.item() * (inp.size(1)-1)
            total_tok += (inp.size(1)-1)

        ppl = math.exp(total_nll / total_tok)
        return ppl

    # --- caching PPL su stati ipotetici (LRU) ---
    def _state_key(self, state: torch.Tensor) -> bytes:
        return state.detach().to('cpu', copy=False).contiguous().numpy().tobytes()

    def _ensure_cache(self, cap=20000):
        if not hasattr(self, "_ppl_cache"):
            self._ppl_cache = OrderedDict()
            self._ppl_cache_cap = cap

    def clear_ppl_cache(self):
        if hasattr(self, "_ppl_cache"):
            self._ppl_cache.clear()

    @torch.no_grad()
    def compute_ppl_for_state(self, state: torch.Tensor, batch_size: int = 64) -> float:
        self._ensure_cache()
        k = self._state_key(state)
        if k in self._ppl_cache:
            v = self._ppl_cache.pop(k)     # move-to-end
            self._ppl_cache[k] = v
            return v

        # snapshot
        saved = [g.alpha.data.clone() for g in self.gates]
        try:
            for gid, gate in enumerate(self.gates):
                gate.alpha.data.fill_(float(state[gid].item()))
            ppl = self.compute_ppl(batch_size=batch_size)
        finally:
            for gate, val in zip(self.gates, saved):
                gate.alpha.data.copy_(val)

        # insert LRU
        self._ppl_cache[k] = float(ppl)
        if len(self._ppl_cache) > self._ppl_cache_cap:
            self._ppl_cache.popitem(last=False)  # drop oldest
        return float(ppl)

    # --- plot utils ---
    @torch.no_grad()
    def plot_scacchiera(self, fname="gate_state.png"):
        import matplotlib.pyplot as plt
        n_layers = self.state.numel() // 2
        mat = self.state.view(n_layers, 2).cpu().numpy()
        fig, ax = plt.subplots(figsize=(4, n_layers * 0.35 + 1.5))
        im = ax.imshow(mat, cmap=plt.cm.get_cmap("Greys", 2), vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["MHA", "FFN"])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_title("Gate state (1=ON, 0=OFF)")
        import matplotlib
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"🔖  plot salvato in → {fname}")

    @torch.no_grad()
    def plot_reward_history(self, fname: str = "reward_history.png"):
        import matplotlib.pyplot as plt
        if not self.reward_history:
            print("[warn] reward_history è vuota – grafico non generato")
            return
        steps = range(1, len(self.reward_history) + 1)
        plt.figure(figsize=(8, 4))
        plt.plot(steps, self.reward_history, lw=2, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Reward ad ogni step dell'episodio")
        plt.grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"📈  curva reward salvata in → {fname}")
