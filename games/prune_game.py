# prune_game.py
import types
from typing import List, Tuple
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import math
from utils import load_model, load_dataset
from .patches import _patch_gpt2_block, _patch_llama_block, ResidualGate


TOGGLE     = 0   # commuta un gate      (arg = gate_id)
SKIP_BLOCK = 1   # spegne MHA+FFN layer (arg = layer_id)
NO_RES     = 2   # spegne solo residuo  (arg = layer_id)
PASS       = 3   # no-op
MIN_LOG = -100.0  

class PruneGame:
    @staticmethod
    def _ensure_patched(model: nn.Module):
        for blk in model.modules():
            if hasattr(blk, "attn") and hasattr(blk, "mlp") and hasattr(blk, "ln_1"):
                _patch_gpt2_block(blk)
            elif hasattr(blk, "self_attn") and hasattr(blk, "mlp") and hasattr(blk, "input_layernorm"):
                _patch_llama_block(blk)
        return model
    
    @staticmethod
    def _collect_gates(model) -> Tuple[torch.Tensor, List[ResidualGate]]:
        ptrs: List[ResidualGate] = []
        for blk in model.modules():
            if all(hasattr(blk, g) for g in ("g_mha", "g_ffn", "g_res")):
                ptrs.extend([blk.g_mha, blk.g_ffn, blk.g_res])
        if not ptrs:
            raise ValueError("Patch failed – no gates found.")
        return torch.ones(len(ptrs), dtype=torch.int8), ptrs

    def __init__(self, args):
        self.args   = args
        self.device = args["device"]

        # model + gates
        self.model_victim = PruneGame._ensure_patched(load_model(args["name_model"], device=self.device, eightbit=args.get("eightbit", False)))
       
        self.model_victim.config.use_cache = False 
        self.state, self.gates = PruneGame._collect_gates(self.model_victim)
        self.state = self.state.to(self.device)
        self.initial_state = self.state.clone()

        # data
        self.tokenizer = self.model_victim.tokenizer
        # n_:samples era a 512
        self.calib_dataset = load_dataset(name=args["name_dataset"], tokenizer=self.tokenizer,split="validation", nsamples=1024, seq_len=128)
        self._cache_reference_logits(batch_size=4)

        # params / targets
        self.target_sparsity = args["target_sparsity"]
        self.tau = args.get("kl_threshold", 0.02) #--> questo se è sotto allora non ha senso
        self.beta = args.get("beta", 3.0)

        # runtime
        self.kl_div = 0.0
        self.time_stamp = 0
        self.R_limit = args.get("R_limit", 120)
        self.history: List[torch.Tensor] = []
        self.state_history = deque(maxlen=self.R_limit-1)
        self.reward = 0.0


    """@torch.no_grad()
    def _cache_reference_logits(self, batch_size: int = 4):
        ref_logits, inputs = [], []
        loader = self.calib_dataset  # already token tensors
        for i in trange(0, len(loader), batch_size, desc="Cache gold logits", leave=False):
            j = min(i + batch_size, len(loader))
            inp = torch.cat(loader[i:j]).long().to(self.device)
            logits = self.model_victim(inp, use_cache=False).logits.detach().cpu()
            ref_logits.append(logits)
            inputs.append(inp.cpu())
        self.ref_logits   = torch.cat(ref_logits)   # N × T × V   (cpu)
        self.calib_inputs = torch.cat(inputs).long().to(self.device)    # N × T       (cpu)"""
    @torch.no_grad()
    def _cache_reference_logits(self, batch_size: int = 4):
        ref_lp, inputs = [], []
        loader = self.calib_dataset
        for i in trange(0, len(loader), batch_size, desc="Cache gold logits", leave=False):
            j   = min(i + batch_size, len(loader))
            inp = torch.cat(loader[i:j]).long().to(self.device)

            # fp32 + log_softmax per evitare overflow/-inf
            logits = self.model_victim(inp, use_cache=False).logits.float()
            lp     = torch.log_softmax(logits, dim=-1).cpu()   # (B,T,V)

            ref_lp.append(lp)
            inputs.append(inp.cpu())

        self.ref_logits   = torch.cat(ref_lp)          # già log-prob
        self.calib_inputs = torch.cat(inputs).long().to(self.device)

    # ------------------------------------------------------------------ gates
    def _toggle_gate(self, gid: int):
        gate = self.gates[gid]
        new = 1 - int(self.state[gid].item())
        gate.alpha.data.fill_(float(new))
        self.state[gid] = new

    def _skip_block(self, layer: int):
        for gid in (layer * 3, layer * 3 + 1):
            if self.state[gid]:
                self._toggle_gate(gid)

    def _no_residual(self, layer: int):
        gid = layer * 3 + 2
        if self.state[gid]:
            self._toggle_gate(gid)

    def _apply_action_in_place(self, action: torch.Tensor):
        gid, op = map(int, action)
        if op == TOGGLE:
            self._toggle_gate(gid)
        elif op == SKIP_BLOCK:
            layer = gid // 3
            self._skip_block(layer)
        elif op == NO_RES:
            layer = gid // 3
            self._no_residual(layer)
        elif op != PASS:
            raise ValueError(op)

    def get_initial_state(self):
        self.state.copy_(self.initial_state)
        for g in self.gates: g.alpha.data.fill_(1.0)
        self.time_stamp = 0
        self.history.clear()
        self.kl_div = 0.0
        self.reward = 0.0
        self.state_history = deque(maxlen=self.R_limit-1)
        return self.state

    def get_next_state(self, state: torch.Tensor, action: torch.Tensor):
        gid, op = map(int, action)
        nxt = state.clone()
        if op == TOGGLE:
            nxt[gid] ^= 1
        elif op == SKIP_BLOCK:
            layer = gid // 3
            nxt[layer*3    ] = 0
            nxt[layer*3 + 1] = 0
        elif op == NO_RES:
            layer = gid // 3
            nxt[layer*3 + 2] = 0
        return nxt
    
    """@torch.no_grad()
    def sparse_incremental_kl(self,batch_size: int = 4,window: int = 1024,penalty: float = 1.5) -> float:
        tau = self.tau                        
        total_kl, total_tok = 0.0, 0
        dev = next(self.model_victim.parameters()).device

        for i in trange(0, len(self.calib_inputs), batch_size, leave=False, desc="Sparse inc-KL"):
            inp = self.calib_inputs[i:i+batch_size].to(dev)          
            ref = self.ref_logits[i:i+batch_size].to(dev)            
            out = self.model_victim(inp, use_cache=False).logits                      

            for t in range(0, inp.size(1)-1, window):
                j = min(t + window, inp.size(1)-1)

                tgt  = inp[:, t:j].unsqueeze(-1)         
                log_p= ref[:, t:j, :].gather(-1, tgt)    
                log_q= out[:, t:j, :].log_softmax(-1).gather(-1, tgt)

                kl_chunk = (log_q - log_p).mean().item() * (-1)
                n_tok    = tgt.numel()

                total_kl  += kl_chunk * n_tok
                total_tok += n_tok

                if total_kl > tau * total_tok:       # early check per uscire
                    return tau * penalty             # oppure total_kl/total_tok

        return total_kl / max(total_tok, 1)          # KL media esatta (≤ τ)"""
    @torch.no_grad()
    def sparse_incremental_kl(self, batch_size: int = 4, window: int = 1024, penalty: float = 1.5) -> float:
        tau       = self.tau
        tot_kl    = 0.0
        tot_tok   = 0
        dev       = next(self.model_victim.parameters()).device

        for i in trange(0, len(self.calib_inputs), batch_size,
                        leave=False, desc="Sparse inc-KL"):
            inp = self.calib_inputs[i:i+batch_size].to(dev)

            ref_lp = self.ref_logits[i:i+batch_size].to(dev)           # log-prob fp32
            out_lp = torch.log_softmax(
                        self.model_victim(inp, use_cache=False)
                            .logits.float(), dim=-1)                   # log-prob fp32

            # --- SOSTITUISCI ±inf CON VALORE FINITO ---------------------------
            ref_lp = torch.where(torch.isfinite(ref_lp), ref_lp,
                                torch.full_like(ref_lp, MIN_LOG))
            out_lp = torch.where(torch.isfinite(out_lp), out_lp,
                                torch.full_like(out_lp, MIN_LOG))

            # ------------------------------------------------------------------
            for t in range(0, inp.size(1)-1, window):
                j   = min(t + window, inp.size(1)-1)
                tgt = inp[:, t:j].unsqueeze(-1)                        # (B,L,1)

                log_p = ref_lp[:, t:j, :].gather(-1, tgt)              # (B,L,1)
                log_q = out_lp[:, t:j, :].gather(-1, tgt)

                kl_chunk = (log_q - log_p).mean().item() * (-1)
                n_tok    = tgt.numel()

                tot_kl  += kl_chunk * n_tok
                tot_tok += n_tok
                if tot_kl > tau * tot_tok:       # early-stop
                    return tau * penalty

        return tot_kl / max(tot_tok, 1)


    """@torch.no_grad()
    def perform_action(self, action: torch.Tensor):
        self._apply_action_in_place(action)
        self.time_stamp += 1
        self.history.append(action.clone())

        if action[1].item() != PASS:
            self.kl_div = self.sparse_incremental_kl()

        sparsity = 1.0 - self.state.float().mean().item()
        self.reward = sparsity - self.beta * self.kl_div

        self.state_history.appendleft(self.state.clone())
        return self.state"""
    
    def perform_action(self, action: torch.Tensor):
        # -- stato pre-mossa
        sparsity_before = 1.0 - self.state.float().mean().item()
        kl_before       = self.kl_div
        ϕ_before        = sparsity_before - self.beta * kl_before

        # -- applica mossa
        self._apply_action_in_place(action)
        self.time_stamp += 1
        self.history.append(action.clone())

        if action[1].item() != PASS:          # ricalcola KL solo se è cambiato qualcosa
            self.kl_div = self.sparse_incremental_kl()

        sparsity_after = 1.0 - self.state.float().mean().item()
        ϕ_after        = sparsity_after - self.beta * self.kl_div
        step_reward    = ϕ_after - ϕ_before   # delta obiettivo
        # -- PASS penalty adattiva ------------------------------------
        if action[1].item() == PASS:
            near_goal = (self.kl_div <= 1.2 * self.tau) and \
                        (sparsity_after >= 0.9 * self.target_sparsity)
            if not near_goal:
                consec_pass = sum(a[1].item() == PASS for a in self.history[-4:])
                step_reward -= 0.5 * (1 + consec_pass)

        self.reward += step_reward

        if self.time_stamp % 3 == 0:   # ogni 3 mosse Δs={Δs:.4f}  Δk={Δk:.4f}
            #print(f"  step{self.time_stamp:2d} r_step={step_reward:.4f}  R_tot={self.reward:.4f}")
            pass
        
        self.state_history.appendleft(self.state.clone())
        return self.state


    def get_scalar(self):
        return torch.tensor([self.R_limit - self.time_stamp], dtype=torch.float32, device=self.device)

    def check_win(self, state):
        sparsity = 1.0 - state.float().mean().item()
        return sparsity >= self.target_sparsity and self.kl_div <= self.tau #abbiamo vinto se abbiamo raggiunto la sparsity e la kl_div ottimale

    def get_value_and_terminated(self, state, node_num_parents=None):
        done = self.check_win(state) or \
            (self.time_stamp >= self.R_limit if node_num_parents is None
                                                else node_num_parents >= self.R_limit)
        return self.reward, done

    def get_encoded_state(self, state: torch.Tensor):

        T, N = self.R_limit, state.numel()
        enc  = torch.zeros((T, N), dtype=torch.float32, device=self.device)

        # stato corrente
        enc[0] = state.float()
        # il resto è la nostra storia
        for t, past_state in enumerate(self.state_history, start=1):
            if t >= T:
                break
            enc[t] = past_state.float()

        return enc

    # evaluatore esterno
    @torch.no_grad()
    def evaluate_new_model(self):
        return self.sparse_incremental_kl()
    
    @torch.no_grad()
    def compute_perplexity(self, full_eval: bool = False, batch_size: int = 4) -> float:
        """
        Calcola la perplexity su `self.calib_dataset`.
        Se `full_eval` è False usa lo stato corrente (potato);
        se True ricarica la maschera iniziale = modello intero.
        """
        if full_eval:
            # Riattivo tutte le porte
            for g in self.gates:
                g.alpha.data.fill_(1.0)
            self.state.fill_(1)

        self.model_victim.eval()
        total_nll, total_tok = 0.0, 0

        loader = self.calib_dataset          # lista di tensor già tokenizzati
        dev = self.device
        for i in range(0, len(loader), batch_size):
            j = min(i + batch_size, len(loader))
            inp = torch.cat(loader[i:j]).long().to(dev)

            # label = input stesso (language modelling causale)
            outputs = self.model_victim(inp, labels=inp, use_cache=False)
            loss = outputs.loss               # media per token sul batch
            total_nll += loss.item() * inp.numel()
            total_tok += inp.numel()

        ppl = math.exp(total_nll / total_tok)
        return ppl
    

    @torch.no_grad()
    def plot_gate_state(self, fname="gate_state.png"):
        import matplotlib.pyplot as plt
        n_layers = self.state.numel() // 3
        mat = self.state.view(n_layers, 3).cpu().numpy()

        fig, ax = plt.subplots(figsize=(4, n_layers * 0.35 + 1.5))
        im = ax.imshow(mat, cmap=plt.cm.get_cmap("Greys", 2), vmin=0, vmax=1)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["MHA", "FFN", "RES"])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_title("Gate state (1=ON, 0=OFF)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"🔖  plot salvato in → {fname}")
