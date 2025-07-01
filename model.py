import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
#  Transformer-based policy / value network per PruneGame
# -----------------------------------------------------------------------------
#   • N = 3×L token (gate)  → stato per layer {MHA, FFN, RES}
#   • encoded_state: (B, T, N)  /  scalars: (B, 1)  (es. passi rimasti)
#   • Ogni token embedding = gate_id_emb + gate_state_emb + scalar_ctx_emb
#   • Heads:
#       – policy: Linear(d, num_ops) per token
#       – value : mean-pool + MLP
# -----------------------------------------------------------------------------

class PruneModel(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        history_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        num_ops: int = 4,
    ):
        super().__init__()
        self.num_blocks = num_blocks          # N token
        self.num_ops    = num_ops
        self.d_model    = d_model

        # ------------------------------------------------ token embeddings
        self.gate_emb   = nn.Embedding(num_blocks, d_model)  # id fissi 0..N-1
        self.state_emb  = nn.Linear(1, d_model)              # stato 0/1
        self.scalar_emb = nn.Linear(1, d_model)              # contesto globale

        # ------------------------------------------------ encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ------------------------------------------------ heads
        self.policy_head = nn.Linear(d_model, num_ops)
        self.value_head  = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, 1),
        )

        self._init_weights()

    # ------------------------------------------------ weights init
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ------------------------------------------------ forward
    def forward(self, encoded_state: torch.Tensor, scalars: torch.Tensor):
        """
        encoded_state: (B, T, N)  float ∈{0,1}
        scalars      : (B, 1)     float   (es. mosse rimaste)
        ritorna: logits (B, N, num_ops) , value (B,)
        """
        B, T, N = encoded_state.shape
        assert N == self.num_blocks, "Mismatch board size"

        # -------- stato corrente (media sui frame)
        state_now = encoded_state.mean(dim=1)               # (B, N)

        # -------- embeddings token
        gate_ids   = torch.arange(N, device=encoded_state.device)   # (N,)
        token_emb  = self.gate_emb(gate_ids)                        # (N, d)
        token_emb  = token_emb.unsqueeze(0).expand(B, N, -1)        # (B,N,d)

        state_emb  = self.state_emb(state_now.unsqueeze(-1))        # (B,N,d)

        # -------- embedding contesto globale scalars
        ctx        = self.scalar_emb(scalars)                       # (B,d)
        ctx        = ctx.unsqueeze(1)                               # (B,1,d)
        x          = token_emb + state_emb + ctx                    # broadcast

        # -------- transformer encoder
        x = self.encoder(x)                                         # (B,N,d)

        # -------- heads
        logits = self.policy_head(x)                                # (B,N,ops)
        value  = self.value_head(x.mean(dim=1)).squeeze(-1)         # (B,)

        return logits, value

    # ------------------------------------------------ train step
    def fwd_train(
        self,
        states : torch.Tensor,   # (B,T,N)
        scalars: torch.Tensor,   # (B,1)
        pi     : torch.Tensor,   # (B, N*ops)
        returns: torch.Tensor,   # (B,1)
        lambda_H: float = 0.02   # <-- bonus entropia (default 0.02)
    ):
        B = states.size(0)
        logits, value = self.forward(states, scalars)        # (B,N,ops)
        logits_flat   = logits.view(B, -1)                   # (B,N*ops)

        # --- policy loss -----------------------------------
        log_probs = F.log_softmax(logits_flat, dim=-1)
        probs     = log_probs.exp()
        pol_loss  = -(pi * log_probs).sum(dim=1).mean()

        # --- entropy bonus ---------------------------------
        ent_loss  = -(probs * log_probs).sum(dim=1).mean()   # = –H

        # --- value loss ------------------------------------
        val_loss  = F.mse_loss(value, returns.squeeze(-1))

        total_loss = pol_loss + val_loss + lambda_H * ent_loss
        return total_loss, pol_loss.detach(), val_loss.detach(), ent_loss.detach()


    # ------------------------------------------------ inference (top-k)
    @torch.no_grad()
    def fwd_infer(self,
                  states : torch.Tensor,  # (B,T,N)
                  scalars: torch.Tensor,  # (B,1)
                  top_k  : int = 64):
        logits, value = self.forward(states, scalars)               # (B,N,ops)
        B, N, _ = logits.shape
        priors   = torch.softmax(logits.view(B, -1), dim=-1)        # (B,N*ops)

        K      = min(top_k, priors.size(-1))
        top_p, top_idx = torch.topk(priors, k=K, dim=-1)            # (B,K)

        block_idx = (top_idx // self.num_ops).long()                # (B,K)
        op_idx    = (top_idx  % self.num_ops).long()                # (B,K)
        actions   = torch.stack([block_idx, op_idx], dim=-1)        # (B,K,2)

        return actions, top_p, value
