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
from utils_datasets import build_calib_dataset
import random


TOGGLE = 0
GATES_PER_LAYER = 2
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
    def _collect_gates(model):
        ptrs = []
        for blk in model.modules():
            if hasattr(blk, "g_mha") and hasattr(blk, "g_ffn"):
                ptrs.extend([blk.g_mha, blk.g_ffn])   # sempre due per layer
        if not ptrs:
            raise ValueError("Patch failed – no gates found.")
        state = torch.ones(len(ptrs), dtype=torch.int8)
        return state, ptrs


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
        #self.calib_dataset = load_dataset(name=args["name_dataset"], tokenizer=self.tokenizer,split="validation", nsamples=100, seq_len=128)
        self.calib_dataset = build_calib_dataset(args["name_dataset"], self.tokenizer, split="validation", nsamples=100, seq_len=128)
        self._cache_reference_logits(batch_size=4)

        # params / targets
        self.target_sparsity = args["target_sparsity"]
        self.tau = args.get("kl_threshold", 0.02) #--> questo se è sotto allora non ha senso
        self.beta = args.get("beta", 3.0)

        # runtime
        self.kl_div = 10_000
        self.time_stamp = 0
        self.R_limit = args.get("R_limit", 120)
        self.history: List[torch.Tensor] = []
        self.state_history = deque(maxlen=self.R_limit-1)
        self.reward = 0.0

        self.fail_penalty = args.get("fail_penalty", 5.0)


    @torch.no_grad()
    def _cache_reference_logits(self, batch_size: int = 4):
        ref_lp, inputs = [], []
        for i in trange(0, len(self.calib_dataset), batch_size, desc="Cache gold logits", leave=False):
            j   = min(i + batch_size, len(self.calib_dataset))
            inp = torch.stack(self.calib_dataset[i:j], dim=0).long().to(self.device)
            logits = self.model_victim(inp, use_cache=False).logits.float()
            lp = torch.log_softmax(logits, dim=-1).cpu()   # (B,L,V)
            ref_lp.append(lp)
            inputs.append(inp.cpu())

        self.ref_logits   = torch.cat(ref_lp)          # log-prob di riferimento
        self.calib_inputs = torch.cat(inputs).long().to(self.device)

    def _toggle_gate(self, gid: int):
        gate = self.gates[gid]
        new = 1 - int(self.state[gid].item())
        gate.alpha.data.fill_(float(new))
        self.state[gid] = new

    def _skip_block(self, layer: int):
        for gid in range(layer*GATES_PER_LAYER, (layer+1)*GATES_PER_LAYER):
            if self.state[gid]:
                self._toggle_gate(gid)

    def _apply_action_in_place(self, action: torch.Tensor):
        gid, op = map(int, action)
        if op == TOGGLE:
            self._toggle_gate(gid)
        else:                    
            pass
    
    def get_initial_state(self):
        self.state.copy_(self.initial_state)
        for g in self.gates:
            g.alpha.data.fill_(1.0)
        
        self.time_stamp   = 0
        self.history.clear()
        self.kl_div       = 0.0
        self.reward       = 0.0
        self.state_history = deque(maxlen=self.R_limit - 1)
        self.consec_pass  = 0
        return self.state

    def get_next_state(self, state: torch.Tensor, action: torch.Tensor):
        gid, op = map(int, action)
        nxt = state.clone()
        if op == TOGGLE:
            nxt[gid] ^= 1
        return nxt
        
    @torch.no_grad()
    def sparse_incremental_kl(self, batch_size: int = 4, window: int = 1024, penalty: float = 1.5,sample_frac: float = 0.25) -> float:

        tau = self.tau
        total_kl, total_tok = 0.0, 0
        dev = next(self.model_victim.parameters()).device

        # --- Logica per il sotto-campionamento (Sampling) ---
        num_total_batches = len(self.calib_inputs) // batch_size
        # Assicurati di campionare almeno un batch
        num_batches_to_sample = max(1, int(num_total_batches * sample_frac))

        all_batch_indices = list(range(num_total_batches))
        sampled_indices = random.sample(all_batch_indices, num_batches_to_sample)
        # --- Fine della logica di sampling ---

        # Itera solo sui batch campionati casualmente
        for batch_idx in sampled_indices:
            i = batch_idx * batch_size
            j = i + batch_size

            # Carica il batch corrente di input e log-prob di riferimento
            inp = self.calib_inputs[i:j].to(dev)
            ref_lp = self.ref_logits[i:j].to(dev)
            
            # Calcola le log-prob del modello potato
            out_logits = self.model_victim(inp, use_cache=False).logits.float()
            out_lp = torch.log_softmax(out_logits, dim=-1)

            # Gestisci i valori ±inf per evitare NaN nel calcolo della KL
            ref_lp = torch.nan_to_num(ref_lp, nan=MIN_LOG, posinf=MIN_LOG, neginf=MIN_LOG)
            out_lp = torch.nan_to_num(out_lp, nan=MIN_LOG, posinf=MIN_LOG, neginf=MIN_LOG)

            # Calcola la KL per finestre di token per gestire la memoria
            for t in range(0, inp.size(1) - 1, window):
                win_end = min(t + window, inp.size(1) - 1)
                target_tokens = inp[:, t:win_end].unsqueeze(-1)

                # Raccogli le log-prob per i token target
                log_p = ref_lp[:, t:win_end, :].gather(-1, target_tokens)
                log_q = out_lp[:, t:win_end, :].gather(-1, target_tokens)

                # Calcola la KL divergence per il chunk corrente
                kl_chunk = (log_p - log_q).mean().item() # KL(P || Q)
                n_tok = target_tokens.numel()

                total_kl += kl_chunk * n_tok
                total_tok += n_tok
                
                # Controllo per l'uscita anticipata se la KL supera la soglia
                if (total_kl / max(total_tok, 1)) > tau:
                    return tau * penalty

        # Ritorna la KL media calcolata sul sotto-campione
        return total_kl / max(total_tok, 1)
    
    #def perform_action(self, action: torch.Tensor):
    #    # -- stato pre-mossa
    #    sparsity_before = 1.0 - self.state.float().mean().item()
    #    kl_before  = self.kl_div
    #    ϕ_before  = sparsity_before - self.beta * kl_before

    #    # -- applica mossa
    #    self._apply_action_in_place(action)
    #    self.time_stamp += 1
    #    self.history.append(action.clone())

    #    
    #    self.kl_div = self.sparse_incremental_kl()

    #    sparsity_after = 1.0 - self.state.float().mean().item()
    #    ϕ_after        = sparsity_after - self.beta * self.kl_div
    #    step_reward    = ϕ_after - ϕ_before   # delta obiettivo

    #    self.reward += step_reward
    #    
    #    self.state_history.appendleft(self.state.clone())
    #    return self.state
    
    def perform_action(self, action):
        self._apply_action_in_place(action)
        self.time_stamp += 1
        self.history.append(action.clone())

        self.kl_div = self.sparse_incremental_kl()
        self.state_history.appendleft(self.state.clone())
        return self.state      


    def get_scalar(self):
        return torch.tensor([self.R_limit - self.time_stamp], dtype=torch.float32, device=self.device)

    def check_win(self, state):
        sparsity = 1.0 - state.float().mean().item()
        return sparsity >= self.target_sparsity and self.kl_div <= self.tau #abbiamo vinto se abbiamo raggiunto la sparsity e la kl_div ottimale
    
    def _state_value(self, state):
        sparsity = 1.0 - state.float().mean().item()
        phi = sparsity - self.beta * self.kl_div
        return phi

    def get_value_and_terminated(self, state, node_num_parents=None):
        win   = self.check_win(state)
        limit = self.time_stamp >= self.R_limit if node_num_parents is None \
                                           else node_num_parents >= self.R_limit
        done  = win or limit
        if done and not win:
            self.reward -= self.fail_penalty       # shaping
        return self._state_value(state), done


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
            #inp = torch.cat(loader[i:j]).long().to(dev)
            inp = torch.stack(loader[i:j], dim=0).long().to(dev)

            outputs = self.model_victim(inp, labels=inp, use_cache=False)
            loss    = outputs.loss.float()           # già shiftata
            total_nll += loss.item() * (inp.size(1)-1)
            total_tok += (inp.size(1)-1)

        ppl = math.exp(total_nll / total_tok)
        return ppl
    

    @torch.no_grad()
    def plot_gate_state(self, fname="gate_state.png"):
        import matplotlib.pyplot as plt
        n_layers = self.state.numel() // GATES_PER_LAYER
        mat = self.state.view(n_layers, GATES_PER_LAYER).cpu().numpy()
        fig, ax = plt.subplots(figsize=(4, n_layers * 0.35 + 1.5))
        im = ax.imshow(mat, cmap=plt.cm.get_cmap("Greys", 2), vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["MHA", "FFN"])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_title("Gate state (1=ON, 0=OFF)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"🔖  plot salvato in → {fname}")
