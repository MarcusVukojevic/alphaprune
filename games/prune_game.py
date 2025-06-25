# prune_game.py
import types
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import math
from utils import load_model, load_dataset

# =============================================================================
#  ACTION SPACE  –  flat: tensor([idx, op])
# =============================================================================
TOGGLE     = 0   # commuta un gate      (arg = gate_id)
SKIP_BLOCK = 1   # spegne MHA+FFN layer (arg = layer_id)
NO_RES     = 2   # spegne solo residuo  (arg = layer_id)
PASS       = 3   # no-op

# =============================================================================
#  ResidualGate – scalare α learnable
# =============================================================================
class ResidualGate(nn.Module):
    def __init__(self, init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, x):
        return self.alpha * x

# =============================================================================
#  Patch helpers
# =============================================================================
def _patch_gpt2_block(block: nn.Module):
    if all(hasattr(block, g) for g in ("g_mha", "g_ffn", "g_res")):
        return
    block.g_mha, block.g_ffn, block.g_res = ResidualGate(), ResidualGate(), ResidualGate()

    ln1, ln2, attn, mlp = block.ln_1, block.ln_2, block.attn, block.mlp

    def fwd(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
            encoder_hidden_states=None, encoder_attention_mask=None,
            use_cache=False, output_attentions=False, **kw):
        residual = hidden_states
        # ---- attention ----------------------------------------------------
        hidden_states_ln = ln1(hidden_states)
        attn_outputs = attn(
            hidden_states_ln,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_out, present = attn_outputs[:2]
        hidden_states = block.g_res(residual) + block.g_mha(attn_out)
        # ---- feed-forward -------------------------------------------------
        residual = hidden_states
        hidden_states_ln = ln2(hidden_states)
        mlp_out = mlp(hidden_states_ln)
        hidden_states = block.g_res(residual) + block.g_ffn(mlp_out)
        return (hidden_states, present) + (() if not output_attentions else attn_outputs[2:3])

    block.forward = types.MethodType(fwd, block)


def _patch_llama_block(block: nn.Module):
    if all(hasattr(block, g) for g in ("g_mha", "g_ffn", "g_res")):
        return
    block.g_mha, block.g_ffn, block.g_res = ResidualGate(), ResidualGate(), ResidualGate()

    sa, mlp = block.self_attn, block.mlp
    ln_in, ln_post = block.input_layernorm, block.post_attention_layernorm

    def fwd(self, hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=False, **kw):
        residual = hidden_states
        attn_out = sa(
            ln_in(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )[0]
        hidden_states = hidden_states + block.g_mha(attn_out)
        hidden_states = block.g_res(residual) + block.g_ffn(mlp(ln_post(hidden_states)))
        return (hidden_states,)

    block.forward = types.MethodType(fwd, block)

# =============================================================================
#  PruneGame
# =============================================================================
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
        self.model_victim = PruneGame._ensure_patched(load_model(args["name_model"], device=self.device, eightbit=args.get("eightbit", False))
                                                      )
        self.state, self.gates = PruneGame._collect_gates(self.model_victim)
        self.state = self.state.to(self.device)
        self.initial_state = self.state.clone()

        # data
        self.tokenizer = self.model_victim.tokenizer
        # n_:samples era a 512
        self.calib_dataset = load_dataset(name=args["name_dataset"], tokenizer=self.tokenizer,split="validation", nsamples=10, seq_len=128)
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
        self.reward = 0.0

    # --------------------------- reference logits (gold, never touched) -----
    @torch.no_grad()
    def _cache_reference_logits(self, batch_size: int = 4):
        ref_logits, inputs = [], []
        loader = self.calib_dataset  # already token tensors
        for i in trange(0, len(loader), batch_size, desc="Cache gold logits", leave=False):
            j = min(i + batch_size, len(loader))
            inp = torch.cat(loader[i:j]).long().to(self.device)
            logits = self.model_victim(inp).logits.detach().cpu()
            ref_logits.append(logits)
            inputs.append(inp.cpu())
        self.ref_logits   = torch.cat(ref_logits)   # N × T × V   (cpu)
        self.calib_inputs = torch.cat(inputs).long().to(self.device)    # N × T       (cpu)

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

    # ------------------------------------------------------------------ env-API
    def get_initial_state(self):
        self.state.copy_(self.initial_state)
        for g in self.gates: g.alpha.data.fill_(1.0)
        self.time_stamp = 0
        self.history.clear()
        self.kl_div = 0.0
        self.reward = 0.0
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

    # ---------------------- KL-divergence incrementale (EvoPress-style) ------
    @torch.no_grad()
    def _compute_incremental_kl(self, batch_size: int = 4) -> float:
        total_kl, total_tok = 0.0, 0
        seq_len = self.calib_inputs.size(1)
        dev = next(self.model_victim.parameters()).device
        for i in range(0, len(self.calib_inputs), batch_size):
            j = min(i + batch_size, len(self.calib_inputs))
            inp = self.calib_inputs[i:j].to(dev)
            logits_p = self.model_victim(inp).logits  # B×T×V
            logits_q = self.ref_logits[i:j].to(dev)   # gold (cpu→dev)

            cum_kl = 0.0
            for t in range(seq_len-1):                # skip last token
                p = torch.log_softmax(logits_q[:, t], dim=-1)
                q = torch.log_softmax(logits_p[:, t], dim=-1)
                kl_t = F.kl_div(q, p, log_target=True, reduction="batchmean")
                cum_kl += kl_t.item()
                total_kl += kl_t.item()
                total_tok += 1
                if cum_kl > self.tau:                 # early-abort
                    break
        return total_kl / max(total_tok, 1)
    

    @torch.no_grad()
    def perform_action(self, action: torch.Tensor):
        self._apply_action_in_place(action)
        self.time_stamp += 1
        self.history.append(action.clone())

        if action[1].item() != PASS:
            self.kl_div = self._compute_incremental_kl()

        sparsity = 1.0 - self.state.float().mean().item()
        self.reward = sparsity - self.beta * self.kl_div
        return self.state


    def get_scalar(self):
        return torch.tensor([self.R_limit - self.time_stamp],
                            dtype=torch.float32, device=self.device)

    def check_win(self, state):
        sparsity = 1.0 - state.float().mean().item()
        return sparsity >= self.target_sparsity and self.kl_div <= self.tau

    def get_value_and_terminated(self, state, node_num_parents=None):
        if node_num_parents is None:                     # real game
            done = self.check_win(state) or self.time_stamp >= self.R_limit
            #print(done)
            return (self.reward if done else 0.0), done
        else:                                            # rollout
            done = self.check_win(state) or node_num_parents >= self.R_limit
            #print(done)
            return (0.0 if self.check_win(state) else -1.0) if done else 0.0, done

    def get_encoded_state(self, state: torch.Tensor):
        # per AlphaZero: (T,N) stack – qui usiamo un dummy repeat
        return state.unsqueeze(0).repeat(self.R_limit, 1).float().to(self.device)

    # evaluatore esterno
    @torch.no_grad()
    def evaluate_new_model(self):
        return self._compute_incremental_kl()
    
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
            outputs = self.model_victim(inp, labels=inp)
            loss = outputs.loss               # media per token sul batch
            total_nll += loss.item() * inp.numel()
            total_tok += inp.numel()

        ppl = math.exp(total_nll / total_tok)
        return ppl
