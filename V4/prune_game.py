import math
import threading
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
        self.eval_mode = args["eval_mode"]  # "ppl" | "kl" | "mse"
        self.args = args

        # --- nuovi flag controllo costi ---
        self.value_uses_surrogate = bool(args.get("value_uses_surrogate", True))
        self.terminal_metric_full = bool(args.get("terminal_metric_full", False))
        self.log_progressive_metric = bool(args.get("log_progressive_metric", False))

        # calib sizes (usati anche dagli stage 'full')
        self.calib_nsamples = int(args.get("calib_nsamples", 5))
        self.calib_seq_len  = int(args.get("calib_seq_len", 128))

        # Tolleranze/parametri consistenti
        # PPL
        self.ppl_tol_frac = args.get("ppl_tol_frac", 0.10)
        self.ppl_alpha    = args.get("ppl_alpha", 0.05)
        # KL
        self.kl_alpha     = args.get("kl_alpha", 0.02)     # scala dell'esponenziale per scoring/reward
        self.kl_tol       = args.get("kl_tol", 0.05)       # soglia "ok" (media per token)
        # MSE (logits)
        self.mse_alpha    = args.get("mse_alpha", 0.02)
        self.mse_tol      = args.get("mse_tol", 0.02)

        # Richiedere anche qualità per early-stop?
        self.require_metric_ok_for_stop = args.get("require_metric_ok_for_stop", False)

        # shaping sparsity
        self.s_beta = args.get("s_beta", 0.05)
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
        self.dataset = build_calib_dataset(
            self.nome_dataset, self.tokenizer,
            split="validation", nsamples=self.calib_nsamples, seq_len=self.calib_seq_len
        )

        # metriche iniziali
        if self.eval_mode == "ppl":
            self.ppl = self.compute_ppl()
            self.initial_ppl = self.ppl
            print("--> ppl iniziale:", self.ppl)
        else:
            # Precalcola teacher logits/logprobs per KL/MSE
            self._cache_teacher_full()
            if self.eval_mode == "kl":
                self.kl = self.compute_kl()
                self.initial_kl = 0.0  # teacher vs teacher = 0
                print("--> kl iniziale (teacher vs teacher): 0.0")
            elif self.eval_mode == "mse":
                self.mse = self.compute_mse()  # MSE logits tra student e teacher (inizio: 0)
                self.initial_mse = 0.0
                print("--> mse iniziale (teacher vs teacher): 0.0")

        # per debug e logging
        self.history = deque(maxlen=self.limite_mosse - 1)
        self.history.appendleft(self.initial_state)
        self.reward_history = []
        self.ppl_history = []
        self.kl_history = []
        self.mse_history = []
        self.last_real_action = None

        # cache LRU per metrica (ppl/kl/mse) + lock per parallelo
        self._metric_cache = OrderedDict()
        self._metric_cache_cap = args.get("metric_cache_cap", 20000)
        self._metric_lock = threading.RLock()

        # baseline ppl per stage (surrogato)
        self._ppl_stage_baseline = {}

    # ---------- Teacher cache ----------
    def _cache_teacher_full(self, batch_size=64, dtype=torch.float16):
        self.model_victim.eval()
        ref_lp, ref_logits, inputs = [], [], []
        with torch.no_grad():
            for i in range(0, len(self.dataset), batch_size):
                j = min(i + batch_size, len(self.dataset))
                inp = torch.stack(self.dataset[i:j], dim=0).long().to(self.device)
                out = self.model_victim(inp)
                logits = out.logits                        # (B,L,V)
                lp = torch.log_softmax(logits, dim=-1)     # (B,L,V)
                ref_lp.append(lp.to(dtype).cpu())
                ref_logits.append(logits.to(dtype).cpu())
                inputs.append(inp.cpu())
        self.teacher_logprobs = torch.cat(ref_lp, dim=0)     # (N,L,V) on CPU fp16
        self.teacher_logits = torch.cat(ref_logits, dim=0)   # (N,L,V) on CPU fp16
        self.calib_inputs_cpu = torch.cat(inputs, dim=0)

    # ---------- Metriche "full" ----------
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
            loss = outputs.loss.float()
            total_nll += loss.item() * (inp.size(1) - 1)
            total_tok += (inp.size(1) - 1)
        ppl = math.exp(total_nll / total_tok)
        return ppl

    @torch.no_grad()
    def compute_kl(self, batch_size: int = 64) -> float:
        """KL(teacher || student) medio su calib set."""
        self.model_victim.eval()
        N = self.teacher_logprobs.size(0)
        tot, count = 0.0, 0
        for i in range(0, N, batch_size):
            j   = min(i + batch_size, N)
            inp = self.calib_inputs_cpu[i:j].to(self.device)
            ref_logp = self.teacher_logprobs[i:j].to(self.device, dtype=torch.float32)  # (B,L,V)
            stu_logp = torch.log_softmax(self.model_victim(inp).logits, dim=-1).float()
            # KL = sum pT * (log pT - log pS)
            kl = (ref_logp.exp() * (ref_logp - stu_logp)).sum(dim=-1).mean().item()
            bsz = (j - i)
            tot += kl * bsz
            count += bsz
        return tot / max(count, 1)

    @torch.no_grad()
    def compute_mse(self, batch_size=64) -> float:
        """MSE tra logits teacher e student."""
        self.model_victim.eval()
        N = self.teacher_logits.size(0)
        tot, count = 0.0, 0
        for i in range(0, N, batch_size):
            j   = min(i + batch_size, N)
            inp = self.calib_inputs_cpu[i:j].to(self.device)
            ref = self.teacher_logits[i:j].to(self.device, dtype=torch.float32)
            stu = self.model_victim(inp).logits.float()
            mse = torch.mean((stu - ref) ** 2).item()
            bsz = (j - i)
            tot += mse * bsz
            count += bsz
        return tot / max(count, 1)

    # ---------- Metriche "subset" (surrogato) ----------
    @torch.no_grad()
    def compute_ppl_subset(self, nsamples: int, seq_len: int, batch_size: int = 64) -> float:
        self.model_victim.eval()
        total_nll, total_tok = 0.0, 0
        N = min(nsamples, len(self.dataset))
        dev = self.device
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            inp = torch.stack(self.dataset[i:j], dim=0).long().to(dev)
            if seq_len < inp.size(1):
                inp = inp[:, :seq_len]
            outputs = self.model_victim(inp, labels=inp, use_cache=False)
            loss = outputs.loss.float()
            total_nll += loss.item() * (inp.size(1) - 1)
            total_tok += (inp.size(1) - 1)
        ppl = math.exp(total_nll / max(total_tok, 1))
        return ppl

    @torch.no_grad()
    def compute_kl_subset(self, nsamples: int, seq_len: int, batch_size: int = 64) -> float:
        self.model_victim.eval()
        N = min(nsamples, self.teacher_logprobs.size(0))
        L_full = self.teacher_logprobs.size(1)
        L = min(seq_len, L_full)
        tot, count = 0.0, 0
        for i in range(0, N, batch_size):
            j   = min(i + batch_size, N)
            inp = self.calib_inputs_cpu[i:j].to(self.device)
            if L < inp.size(1):
                inp = inp[:, :L]
            ref_logp = self.teacher_logprobs[i:j, :L].to(self.device, dtype=torch.float32)  # (B,L,V)
            stu_logp = torch.log_softmax(self.model_victim(inp).logits, dim=-1).float()     # (B,L,V)
            kl = (ref_logp.exp() * (ref_logp - stu_logp)).sum(dim=-1).mean().item()
            bsz = (j - i)
            tot += kl * bsz
            count += bsz
        return tot / max(count, 1)

    @torch.no_grad()
    def compute_mse_subset(self, nsamples: int, seq_len: int, batch_size: int = 64) -> float:
        self.model_victim.eval()
        N = min(nsamples, self.teacher_logits.size(0))
        L_full = self.teacher_logits.size(1)
        L = min(seq_len, L_full)
        tot, count = 0.0, 0
        for i in range(0, N, batch_size):
            j   = min(i + batch_size, N)
            inp = self.calib_inputs_cpu[i:j].to(self.device)
            if L < inp.size(1):
                inp = inp[:, :L]
            ref = self.teacher_logits[i:j, :L].to(self.device, dtype=torch.float32)
            stu = self.model_victim(inp).logits.float()
            mse = torch.mean((stu - ref) ** 2).item()
            bsz = (j - i)
            tot += mse * bsz
            count += bsz
        return tot / max(count, 1)

    # ---------- Cache metrica per stati ipotetici ----------
    def _state_key(self, state: torch.Tensor, stage_tag: str | None = None) -> bytes:
        base = state.detach().to('cpu', copy=False).contiguous().numpy().tobytes()
        if stage_tag is None:
            return base
        return base + str(stage_tag).encode('utf-8')

    def _cache_get(self, k):
        if k in self._metric_cache:
            v = self._metric_cache.pop(k)
            self._metric_cache[k] = v
            return v
        return None

    def _cache_put(self, k, v: float):
        self._metric_cache[k] = float(v)
        if len(self._metric_cache) > self._metric_cache_cap:
            self._metric_cache.popitem(last=False)

    def clear_metric_cache(self):
        self._metric_cache.clear()

    # alias per retro-compat (chiamato nel codice sequenziale esistente)
    def clear_ppl_cache(self):
        self.clear_metric_cache()

    # ---------- helper stage ----------
    def _resolve_stage_spec(self, stage_idx: int | None):
        stages = self.args.get("prog_stages")
        if not stages:
            stages = [("full", "full")]
        idx = 0 if stage_idx is None else max(0, min(int(stage_idx), len(stages) - 1))
        ns, sl = stages[idx]
        if ns == "full":
            ns = self.calib_nsamples
        if sl == "full":
            sl = self.calib_seq_len
        return int(ns), int(sl), idx

    # ---------- FULL metric (per usi “rari”) ----------
    @torch.no_grad()
    def _compute_metric_for_state(self, state: torch.Tensor, batch_size: int = 64) -> float:
        saved = [g.alpha.data.clone() for g in self.gates]
        try:
            for gid, gate in enumerate(self.gates):
                gate.alpha.data.fill_(float(state[gid].item()))
            if self.eval_mode == "ppl":
                return self.compute_ppl(batch_size=batch_size)
            elif self.eval_mode == "kl":
                return self.compute_kl(batch_size=batch_size)
            elif self.eval_mode == "mse":
                return self.compute_mse(batch_size=batch_size)
            else:
                raise ValueError(f"Unknown eval_mode: {self.eval_mode}")
        finally:
            for gate, val in zip(self.gates, saved):
                gate.alpha.data.copy_(val)

    @torch.no_grad()
    def metric_from_state_or_cache(self, state: torch.Tensor) -> float:
        k = self._state_key(state)
        hit = self._cache_get(k)
        if hit is not None:
            return hit
        val = self._compute_metric_for_state(state)
        self._cache_put(k, val)
        return val

    # ---------- SURROGATO (thread-safe) ----------
    @torch.no_grad()
    def evaluate_metric_shared(self, state: torch.Tensor, stage_idx: int | None = None) -> float:
        ns, sl, stg = self._resolve_stage_spec(stage_idx)
        stage_tag = f"stg{stg}:{ns}x{sl}"
        k = self._state_key(state, stage_tag=stage_tag)

        with self._metric_lock:
            hit = self._cache_get(k)
            if hit is not None:
                return hit

            saved = [g.alpha.data.clone() for g in self.gates]
            try:
                for gid, gate in enumerate(self.gates):
                    gate.alpha.data.fill_(float(state[gid].item()))
                if self.eval_mode == "ppl":
                    val = self.compute_ppl_subset(ns, sl)
                elif self.eval_mode == "kl":
                    val = self.compute_kl_subset(ns, sl)
                elif self.eval_mode == "mse":
                    val = self.compute_mse_subset(ns, sl)
                else:
                    raise ValueError(f"Unknown eval_mode: {self.eval_mode}")
            finally:
                for gate, val_alpha in zip(self.gates, saved):
                    gate.alpha.data.copy_(val_alpha)

            self._cache_put(k, float(val))
            return float(val)

    @torch.no_grad()
    def metric_progressive(self, state: torch.Tensor, stage_idx: int | None):
        return self.evaluate_metric_shared(state, stage_idx=stage_idx)

    def get_stage_baseline_for_ppl(self, stage_idx: int) -> float:
        ns, sl, stg = self._resolve_stage_spec(stage_idx)
        key = (stg, ns, sl)
        if key in self._ppl_stage_baseline:
            return self._ppl_stage_baseline[key]
        with torch.no_grad():
            val = self.compute_ppl_subset(ns, sl)
        self._ppl_stage_baseline[key] = float(val)
        return float(val)

    # ---------- Interfaccia AlphaZero ----------
    def reset_game(self):
        # reset stato e gates
        self.state.copy_(self.initial_state)
        for g in self.gates:
            g.alpha.data.fill_(1.0)

        # metriche correnti (solo placeholder, niente full durante self-play)
        if self.eval_mode == "ppl":
            self.ppl = self.initial_ppl
        elif self.eval_mode == "kl":
            self.kl = 0.0
        elif self.eval_mode == "mse":
            self.mse = 0.0

        self.numero_mossa = 0
        self.history = deque(maxlen=self.limite_mosse - 1)
        self.history.appendleft(self.initial_state)
        return self.state

    @torch.no_grad()
    def get_value_and_terminated(self, state, depth=None, register=False):
        """
        Value/terminazione USANDO SURROGATO (veloce).
        PPL full calcolata solo se terminal_metric_full=True e siamo a fine episodio.
        """
        steps = self.numero_mossa if depth is None else depth

        # metriche sullo stato ipotetico (surrogato)
        s = 1.0 - state.float().mean().item()
        target = self.target_sparsity
        stage_idx = 0
        m_now = self.metric_progressive(state, stage_idx=stage_idx)

        # map metrica -> (term weight, tol)
        if self.eval_mode == "ppl":
            init_ppl = self.get_stage_baseline_for_ppl(stage_idx)
            rel   = max(0.0, (m_now - init_ppl) / (init_ppl + 1e-8))
            m_term = math.exp(- rel / (self.ppl_alpha + 1e-12))  # (0,1]
            m_ok   = (rel <= self.ppl_tol_frac)
        elif self.eval_mode == "kl":
            m_term = math.exp(- m_now / (self.kl_alpha + 1e-12))
            m_ok   = (m_now <= self.kl_tol)
        elif self.eval_mode == "mse":
            m_term = math.exp(- m_now / (self.mse_alpha + 1e-12))
            m_ok   = (m_now <= self.mse_tol)
        else:
            raise ValueError

        hit_s = (s >= target)
        hit_q = m_ok if self.require_metric_ok_for_stop else True
        limit = (steps >= self.limite_mosse)

        # EARLY STOP
        if hit_s and hit_q:
            reward = min(1.0, 0.7 + 0.3 * m_term)
            if register:
                self.reward_history.append(float(reward))
            if self.terminal_metric_full and self.eval_mode == "ppl":
                _ = self.metric_from_state_or_cache(state)  # PPL full per logging
            return float(reward), True

        if not limit:
            return 0.0, False

        # Limite mosse senza centratura: reward basso (0..~0.4)
        progress = max(0.0, min(1.0, s / max(target, 1e-8)))
        reward = 0.4 * (progress * (0.5 + 0.5 * m_term))
        if register:
            self.reward_history.append(float(reward))
        if self.terminal_metric_full and self.eval_mode == "ppl":
            _ = self.metric_from_state_or_cache(state)  # PPL full per logging
        return float(reward), True

    # features per la rete
    def get_encoded_state(self, state: torch.Tensor):
        T, N = self.limite_mosse, state.numel()
        enc = torch.zeros((T, N), dtype=torch.float32, device=self.device)
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
        _ = self.disable_gate(gid)
        self.numero_mossa += 1
        self.last_real_action = gid
        self.history.appendleft(self.state.clone())

        # NIENTE full compute qui; opzionale logging surrogato
        if self.log_progressive_metric:
            _ = self.metric_progressive(self.state, stage_idx=0)

        return self.state

    def legal_mask(self, state, is_root: bool, parent_action: int | None):
        mask = state.bool().clone()  # True solo dove state==1
        if is_root and self.last_real_action is not None:
            mask[self.last_real_action] = False
        if parent_action is not None:
            mask[parent_action] = False
        return mask

    def pensa_azione(self, state: torch.Tensor, action: int):
        if int(state[action].item()) == 0:
            return state  # illegal -> invariato
        nuovo_stato = state.clone()
        nuovo_stato[action] = 0
        return nuovo_stato

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
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        import numpy as np  # noqa: F401
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
