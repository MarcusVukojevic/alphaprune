import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
#  Transformer-based policy / value network for PruneGame
# -----------------------------------------------------------------------------
#   • Board has N = 3 × L gate tokens (layer × {MHA, FFN, RES})
#   • Encoded history arrives as (B, T, N) with T frames (R_limit)
#   • We average over the T dimension → (B, N) current state probability
#   • Each gate is a token:   token_emb = gate_id_emb + state_scalar_emb
#   • A small TransformerEncoder maps sequence (N, B, d) → same shape
#   • Heads:
#       – policy: Linear(d, num_ops) per token   → logits (B, N, num_ops)
#       – value : mean-pool tokens → Linear → tanh
# -----------------------------------------------------------------------------

class PruneModel(nn.Module):
    def __init__(
        self,
        num_blocks: int,            # N = 3 × L
        history_len: int,           # R_limit (not used directly)
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        num_ops: int = 4,           # TOGGLE, SKIP_BLOCK, NO_RES, PASS
    ):
        super().__init__()
        self.num_blocks  = num_blocks
        self.num_ops     = num_ops
        self.d_model     = d_model

        # token (gate) id embedding
        self.gate_emb = nn.Embedding(num_blocks, d_model)
        # scalar state 0/1 embedding (linear scale)
        self.state_emb = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # heads
        self.policy_head = nn.Linear(d_model, num_ops)
        self.value_head  = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ fwd
    def forward(self, encoded_state: torch.Tensor):
        """encoded_state: (B, T, N) float  in {0,1}
        returns logits (B, N, num_ops)  and value (B,)
        """
        B, T, N = encoded_state.shape
        assert N == self.num_blocks, "Mismatch board size"

        # aggregate history – simple average over frames
        state_now = encoded_state.mean(dim=1)        # (B, N)

        # token embeddings
        gate_ids = torch.arange(N, device=encoded_state.device)  # (N,)
        token_emb = self.gate_emb(gate_ids)                      # (N, d)
        token_emb = token_emb.unsqueeze(0).expand(B, N, -1)      # (B, N, d)

        state_scalar = state_now.unsqueeze(-1)                   # (B, N, 1)
        state_emb    = self.state_emb(state_scalar)              # (B, N, d)

        x = token_emb + state_emb                                # (B, N, d)

        # Transformer encoder expects (B, N, d) with batch_first=True
        x = self.encoder(x)                                      # (B, N, d)

        # policy logits per token
        logits = self.policy_head(x)                             # (B, N, num_ops)

        # value: mean-pool tokens then head → (B,)
        pooled = x.mean(dim=1)                                   # (B, d)
        value  = self.value_head(pooled).squeeze(-1)             # (B,)
        return logits, value

    # ------------------------------------------------------------- losses
    def fwd_train(self, states: torch.Tensor, scalars: torch.Tensor,
                  actions: torch.Tensor, returns: torch.Tensor):
        """Compute policy cross-entropy + value MSE"""
        B = states.size(0)
        logits, value = self.forward(states)
        logits_flat = logits.view(B, -1)          # (B, N*num_ops)

        block_idx = actions[:, 0].long()
        op_idx    = actions[:, 1].long()
        action_idx = block_idx * self.num_ops + op_idx  # (B,)

        pol_loss = F.cross_entropy(logits_flat, action_idx, reduction="mean")
        val_loss = F.mse_loss(value, returns.view(-1), reduction="mean")
        return pol_loss, val_loss

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def fwd_infer(self, states: torch.Tensor, scalars: torch.Tensor, top_k: int = 64):
        logits, value = self.forward(states)           # logits (B,N,ops)
        B, N, _ = logits.shape
        priors = torch.softmax(logits.view(B, -1), dim=-1)  # (B, N*ops)
        K = min(top_k, priors.size(-1))
        top_vals, top_idx = torch.topk(priors, k=K, dim=-1)

        block_idx = (top_idx // self.num_ops).long()
        op_idx    = (top_idx  % self.num_ops).long()
        actions   = torch.stack([block_idx, op_idx], dim=-1)  # (B,K,2)
        return actions, top_vals, value
