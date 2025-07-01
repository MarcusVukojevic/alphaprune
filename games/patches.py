import types
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
#  ResidualGate – scalare α learnable
# =============================================================================
class ResidualGate(nn.Module):
    def __init__(self, init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, x):
        return self.alpha * x

MIN_LOG = -100.0  
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

    block.g_mha, block.g_ffn, block.g_res = (
        ResidualGate(), ResidualGate(), ResidualGate()
    )

    sa, mlp = block.self_attn, block.mlp
    ln_in, ln_post = block.input_layernorm, block.post_attention_layernorm

    def fwd(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kw,
    ):
        residual = hidden_states

        attn_out, present = sa(                       # <- la MHA originale restituisce
            ln_in(hidden_states),                     #    (attn_out, present_kv, attn_w)
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )[:2]                                         # teniamo solo out e present

        hidden_states = hidden_states + block.g_mha(attn_out)
        hidden_states = block.g_res(residual) + block.g_ffn(
            mlp(ln_post(hidden_states))
        )

        # ------- RISPETTA la firma ufficiale -----------------
        if output_attentions:
            # restituisce anche l’attenzione → serve terzo elem.
            return (hidden_states, present, None)
        else:
            return (hidden_states, present)

    block.forward = types.MethodType(fwd, block)