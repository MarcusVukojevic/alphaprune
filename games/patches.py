# patches.py
import types, torch.nn as nn, torch

class ResidualGate(nn.Module):
    def __init__(self, init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init, dtype=torch.float32))
    def forward(self, x): return self.alpha * x
# ------------------------------------------------------------
def _patch_gpt2_block(blk):
    if hasattr(blk, "g_mha"): return
    blk.g_mha, blk.g_ffn = ResidualGate(), ResidualGate()

    # MHA -----------------------------------------------------
    attn_fn = blk.attn.__class__.forward          # UN-bound!
    def attn_fwd(self, *args, **kw):
        out = attn_fn(self, *args, **kw)          # pass self esplicito
        return (blk.g_mha(out[0]), *out[1:])
    blk.attn.forward = types.MethodType(attn_fwd, blk.attn)

    # FFN -----------------------------------------------------
    mlp_fn = blk.mlp.__class__.forward
    def mlp_fwd(self, *a, **kw):
        return blk.g_ffn(mlp_fn(self, *a, **kw))
    blk.mlp.forward  = types.MethodType(mlp_fwd, blk.mlp)
# ------------------------------------------------------------
def _patch_llama_block(blk):
    if hasattr(blk, "g_mha"): return
    if not (hasattr(blk, "self_attn") and hasattr(blk, "mlp")): return

    blk.g_mha, blk.g_ffn = ResidualGate(), ResidualGate()

    # MHA -----------------------------------------------------
    sa_fn = blk.self_attn.__class__.forward       # UN-bound
    def sa_fwd(self, *args, **kw):
        out = sa_fn(self, *args, **kw)
        return (blk.g_mha(out[0]), *out[1:])
    blk.self_attn.forward = types.MethodType(sa_fwd, blk.self_attn)

    # FFN -----------------------------------------------------
    mlp_fn = blk.mlp.__class__.forward
    def mlp_fwd(self, *a, **kw):
        return blk.g_ffn(mlp_fn(self, *a, **kw))
    blk.mlp.forward = types.MethodType(mlp_fwd, blk.mlp)
