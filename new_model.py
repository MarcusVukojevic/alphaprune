"""
Alphaprune – attention‑powered heads (rev 2, 2025‑07‑11)
=======================================================
Questa versione mantiene **identica** la firma pubblica di `PruneModel`
ma rimpiazza le vecchie _PolicyHead_ e _ValueHead_ lineari con
implementazioni basate su **Multi‑Head Attention**, ispirate al codice
che mi hai passato.

Principali differenze rispetto alla rev precedente:
  • `MultiHeadAttention`, `Head` e helper presi quasi 1‑a‑1 dal tuo
    snippet.
  • **Policy head**: un blocco MHA self‑attention (LayerNorm → MHA →
    residual) + FFN; proietta poi a `num_ops` per token.
  • **Value head**: introduce un token "[V]" learnable che attende su
    tutti i blocchi (cross‑attention) e passa il risultato a un MLP
    quantile‑regression come prima.
  • Nessun cambio a `__init__`, `forward`, `fwd_train`, `fwd_infer`.

Dipendenze: solo `torch`.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# =============================================================================
# Positional‑encoding helper (sinusoidale, non trainabile)
# =============================================================================

def create_fixed_positional_encoding(n_position: int, n_embedding: int, device):
    pe = torch.zeros(n_position, n_embedding, device=device)
    positions = torch.arange(n_position, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, n_embedding, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / n_embedding)
    )
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe

# =============================================================================
# Multi‑Head Attention building blocks (copiati dal tuo esempio)
# =============================================================================

class _Head(nn.Module):
    def __init__(self, c1: int, c2: int, d: int, causal_mask: bool = False):
        super().__init__()
        self.d = d
        self.causal_mask = causal_mask
        self.query = nn.Linear(c1, d, bias=False)
        self.key   = nn.Linear(c2, d, bias=False)
        self.value = nn.Linear(c2, d, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        q = self.query(x)                          # (*, nx, d)
        k = self.key(y)                            # (*, ny, d)
        v = self.value(y)                          # (*, ny, d)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.d)  # (*, nx, ny)
        if self.causal_mask:
            mask = torch.tril(torch.ones_like(att))
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v                              # (*, nx, d)
        return out

class _MultiHeadAttention(nn.Module):
    def __init__(self, c1: int, c2: int, n_heads: int = 8, d_head: Optional[int] = None, ffn_mul: int = 4, causal_mask: bool = False):
        super().__init__()
        d_head = d_head or (c1 // n_heads)
        self.nh = n_heads
        self.dh = d_head
        self.ln_q = nn.LayerNorm(c1)
        self.ln_kv = nn.LayerNorm(c2)
        self.heads = nn.ModuleList([
            _Head(c1, c2, d_head, causal_mask=causal_mask) for _ in range(n_heads)
        ])
        self.proj_out = nn.Linear(n_heads * d_head, c1)
        # FFN post‑attention
        self.ln_ff = nn.LayerNorm(c1)
        self.ff = nn.Sequential(
            nn.Linear(c1, c1 * ffn_mul),
            nn.GELU(),
            nn.Linear(c1 * ffn_mul, c1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_norm = self.ln_q(x)
        y_norm = self.ln_kv(y)
        concat = torch.cat([h(x_norm, y_norm) for h in self.heads], dim=-1)
        x = x + self.proj_out(concat)              # residual 1
        out = x + self.ff(self.ln_ff(x))           # residual 2
        return out

# =============================================================================
# Quantile value head (uguale a prima, cambia solo nome classe public)
# =============================================================================

class _QuantileValueProj(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 512, n_quantiles: int = 8):
        super().__init__()
        self.nq = n_quantiles
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, n_quantiles)
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

    # --- utils ----------------------------------------------------
    @staticmethod
    def risk_adjust(q: torch.Tensor, uq: float = 0.75):
        k = math.floor(uq * q.size(-1))
        return q[..., k:].mean(dim=-1)

    @staticmethod
    def quantile_loss(q_pred: torch.Tensor, y: torch.Tensor, delta: float = 1.0):
        n = q_pred.size(-1)
        taus = (torch.arange(n, device=q_pred.device, dtype=q_pred.dtype) + 0.5) / n
        diff = y.unsqueeze(-1) - q_pred
        huber = F.huber_loss(q_pred, y.unsqueeze(-1).expand_as(q_pred), reduction="none", delta=delta)
        return (torch.abs(taus - (diff < 0).float()) * huber).mean()

# =============================================================================
# Torso (embed + Transformer + FFN) ‑ invariato
# =============================================================================

class _PruneTorso(nn.Module):
    def __init__(self, num_blocks: int, history_len: int, d_model: int, n_heads: int, n_layers: int, dim_feedforward: int):
        super().__init__()
        self.N = num_blocks
        self.C = d_model
        self.gate_emb = nn.Embedding(num_blocks, d_model)
        self.history_emb = nn.Linear(history_len, d_model)
        self.scalar_emb = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.post_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward), nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
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
        assert N == self.N, "Mismatch num_blocks"
        hist_feat = self.history_emb(encoded_state.permute(0, 2, 1))  # (B,N,C)
        id_feat = self.gate_emb(torch.arange(N, device=encoded_state.device).expand(B, -1))
        ctx_feat = self.scalar_emb(scalars).unsqueeze(1)
        x = hist_feat + id_feat + ctx_feat
        x = self.encoder(x)
        x = x + self.post_ffn(x)
        return x  # (B,N,C)

# =============================================================================
# Attention‑powered Policy head
# =============================================================================

class _PolicyHeadAttn(nn.Module):
    def __init__(self, d_model: int, num_ops: int, n_heads: int = 8):
        super().__init__()
        self.mha = _MultiHeadAttention(d_model, d_model, n_heads=n_heads)
        self.proj_logits = nn.Linear(d_model, num_ops)

    def forward(self, emb: torch.Tensor):
        # emb: (B,N,C)
        h = self.mha(emb, emb)              # self‑attn over blocks
        logits = self.proj_logits(h)        # (B,N,num_ops)
        return logits

# =============================================================================
# Attention‑pooled Value head (learnable [V] token)
# =============================================================================

class _ValueHeadAttn(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 512, n_quantiles: int = 8, n_heads: int = 8):
        super().__init__()
        self.v_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mha = _MultiHeadAttention(d_model, d_model, n_heads=n_heads)
        self.proj = _QuantileValueProj(d_model, d_hidden, n_quantiles)
        nn.init.normal_(self.v_token, mean=0.0, std=0.02)

    def forward(self, emb: torch.Tensor):
        B = emb.size(0)
        vtok = self.v_token.expand(B, -1, -1)          # (B,1,C)
        pooled = self.mha(vtok, emb)                   # (B,1,C)
        q = self.proj(pooled.squeeze(1))               # (B,nq)
        return q

# =============================================================================
# Public PruneModel (API invariata)
# =============================================================================

class PruneModel(nn.Module):
    def __init__(self,num_blocks: int,history_len: int,d_model: int = 128,n_heads: int = 4,n_layers: int = 2,dim_feedforward: int = 256,num_ops: int = 2,attn_heads: int = 8,):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_ops = num_ops
        self.torso = _PruneTorso(num_blocks, history_len, d_model, n_heads, n_layers, dim_feedforward)
        self.policy_head = _PolicyHeadAttn(d_model, num_ops, n_heads=attn_heads)
        self.value_head  = _ValueHeadAttn(d_model, d_hidden=dim_feedforward*2, n_quantiles=8, n_heads=attn_heads)

    def forward(self, encoded_state: torch.Tensor, scalars: torch.Tensor):
        emb = self.torso(encoded_state, scalars)            # (B,N,C)
        logits = self.policy_head(emb)                      # (B,N,num_ops)
        quant = self.value_head(emb)                        # (B,nq)
        value = _QuantileValueProj.risk_adjust(quant)       # (B,)
        return logits, value

    def fwd_train(self,states: torch.Tensor,scalars: torch.Tensor,pi: torch.Tensor,returns: torch.Tensor,lambda_H: float = 0.02,):
        B = states.size(0)
        emb = self.torso(states, scalars)
        logits = self.policy_head(emb)
        quant = self.value_head(emb)
        value = _QuantileValueProj.risk_adjust(quant)

        # ---- policy loss
        log_probs = F.log_softmax(logits.view(B, -1), dim=-1)
        pol_loss = -(pi * log_probs).sum(dim=-1).mean()
        probs = log_probs.exp()
        ent_loss = (probs * log_probs).sum(dim=-1).mean()

        # ---- value loss
        val_loss = _QuantileValueProj.quantile_loss(quant, returns.squeeze(-1))
        loss = pol_loss + val_loss - lambda_H * ent_loss
        return loss, pol_loss.detach(), val_loss.detach(), ent_loss.detach()

    @torch.no_grad()
    def fwd_infer(self,states: torch.Tensor,scalars: torch.Tensor,top_k: int = 32,):
        """
        Restituisce:
          • actions  – tensor (B, K, 2)   [block_id, op_id]  (op_id == 0 ⇒ TOGGLE)
          • priors   – tensor (B, K)      prior normalizzati
          • value    – tensor (B,)        stima del valore
        """
        logits, value = self.forward(states, scalars)            # logits (B,N,num_ops)
        priors = torch.softmax(logits.view(states.size(0), -1), dim=-1)  # (B, N·num_ops)

        # ---------------------------------------------------------------------
        # 1) Mantieni solo le azioni TOGGLE  (op_idx == 0)
        # ---------------------------------------------------------------------
        mask_toggle = torch.arange(self.num_ops, device=priors.device) \
                         .repeat(self.num_blocks) == 0            # shape (N·num_ops,)
        priors = priors[:, mask_toggle]                           # (B, N)

        # 2) Dirichlet noise globale per incentivare esplorazione
        alpha = 0.3
        conc = torch.full_like(priors, alpha, device="cpu")       # campiono su CPU
        noise = torch.distributions.Dirichlet(conc).sample().to(priors.device)
        eps = 0.20
        priors = (1.0 - eps) * priors + eps * noise

        # 3) Normalizza di nuovo (potrebbero esserci underflow)
        priors = priors / priors.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # 4) Top-k e costruzione azioni   (op_id è sempre 0 ⇒ TOGGLE)
        K = min(top_k, priors.size(-1))
        top_p, top_idx = torch.topk(priors, k=K, dim=-1)          # idx ∈ [0, N)
        actions = torch.stack(
            [top_idx, torch.zeros_like(top_idx)], dim=-1          # (B,K,2)
        )

        return actions, top_p, value