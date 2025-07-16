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
    def __init__( self, num_blocks: int, history_len: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, dim_feedforward: int = 256, num_ops: int = 3,):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_ops = num_ops
        self.d_model = d_model

        # 1. Embedding per l'ID di ogni porta (0, 1, ..., N-1)
        self.gate_emb = nn.Embedding(num_blocks, d_model)
        
        # 2. Embedding per la storia degli stati (0/1) di ogni porta
        self.history_emb = nn.Linear(history_len, d_model)
        
        # 3. Embedding per il contesto globale (es. passi rimanenti)
        self.scalar_emb = nn.Linear(1, d_model)
        

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # --- Heads ---
        self.policy_head = nn.Linear(d_model, num_ops)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, 1),
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

    def forward(self, encoded_state: torch.Tensor, scalars: torch.Tensor):
        B, T, N = encoded_state.shape
        assert N == self.num_blocks, "Mismatch board size"
        assert T == self.history_emb.in_features, "Mismatch history length"

        # 1. Permuta l'input per avere la storia come feature per ogni porta
        gate_histories = encoded_state.permute(0, 2, 1) # Shape: (B, N, T)

        # 2. Applica l'embedding sulla storia per ottenere le feature di stato
        state_features = self.history_emb(gate_histories) # Shape: (B, N, d_model)

        # 3. Aggiungi l'embedding dell'ID della porta
        gate_ids = torch.arange(N, device=encoded_state.device).expand(B, -1) # (B, N)
        gate_id_features = self.gate_emb(gate_ids) # Shape: (B, N, d_model) <-- NOME CORRETTO
        
        # 4. Aggiungi il contesto scalare
        ctx_features = self.scalar_emb(scalars).unsqueeze(1) # Shape: (B, 1, d_model)

        # 5. Combina le feature per l'input del Transformer
        x = state_features + gate_id_features + ctx_features # Broadcasting su N

        # Passa nel Transformer e negli heads
        x = self.encoder(x)
        logits = self.policy_head(x)
        value = self.value_head(x.mean(dim=1)).squeeze(-1)

        return logits, value

    def fwd_train(
        self,
        states: torch.Tensor,
        scalars: torch.Tensor,
        pi: torch.Tensor,
        returns: torch.Tensor,
        lambda_H: float = 0.02
    ):
        B = states.size(0)
        logits, value = self.forward(states, scalars)
        logits_flat = logits.view(B, -1)

        log_probs = F.log_softmax(logits_flat, dim=-1)
        pol_loss = -(pi * log_probs).sum(dim=-1).mean()
        
        probs = log_probs.exp()
        ent_loss = (probs * log_probs).sum(dim=-1).mean() # Nota: l'entropia è negativa, la loss è -H

        val_loss = F.mse_loss(value, returns.squeeze(-1))

        total_loss = pol_loss + val_loss - (lambda_H * ent_loss) # Sottraggo perché ent_loss = -H
        return total_loss, pol_loss.detach(), val_loss.detach(), ent_loss.detach()

    @torch.no_grad()
    def fwd_infer(
        self,
        states: torch.Tensor,
        scalars: torch.Tensor,
        top_k: int = 32
    ):
        logits, value = self.forward(states, scalars)
        B, N, _ = logits.shape
        priors = torch.softmax(logits.view(B, -1), dim=-1)

        K = min(top_k, priors.size(-1))
        top_p, top_idx = torch.topk(priors, k=K, dim=-1)

        block_idx = (top_idx // self.num_ops).long()
        op_idx = (top_idx % self.num_ops).long()
        actions = torch.stack([block_idx, op_idx], dim=-1)

        return actions, top_p, value