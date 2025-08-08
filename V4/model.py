import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Head(nn.Module):
    def __init__(self, c1: int, c2: int, d: int, causal_mask: bool = False):
        super().__init__()
        self.d = d
        self.causal_mask = causal_mask
        self.query = nn.Linear(c1, d, bias=False)
        self.key = nn.Linear(c2, d, bias=False)
        self.value = nn.Linear(c2, d, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        q = self.query(x)
        k = self.key(y)
        v = self.value(y)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        if self.causal_mask:
            mask = torch.tril(torch.ones_like(att))
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v
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
        x = x + self.proj_out(concat)
        out = x + self.ff(self.ln_ff(x))
        return out


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


class _PolicyHeadAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.mha = _MultiHeadAttention(d_model, d_model, n_heads=n_heads)
        self.proj_logits = nn.Linear(d_model, 1)  # 1 logit per gate

    def forward(self, emb: torch.Tensor):
        h = self.mha(emb, emb)            # self-attn over blocks
        logits = self.proj_logits(h).squeeze(-1)   # (B, N)
        return logits


class _ValueHeadAttn(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 512, n_quantiles: int = 8, n_heads: int = 8):
        super().__init__()
        self.v_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mha = _MultiHeadAttention(d_model, d_model, n_heads=n_heads)
        self.proj = _QuantileValueProj(d_model, d_hidden, n_quantiles)
        nn.init.normal_(self.v_token, mean=0.0, std=0.02)

    def forward(self, emb: torch.Tensor):
        B = emb.size(0)
        vtok = self.v_token.expand(B, -1, -1)  # (B,1,C)
        pooled = self.mha(vtok, emb)           # (B,1,C)
        q = self.proj(pooled.squeeze(1))       # (B,nq)
        return q


class PruneModel(nn.Module):
    def __init__(self, num_blocks, history_len, d_model=128, n_heads=4, n_layers=2, dim_feedforward=256, attn_heads=8):
        super().__init__()
        self.num_blocks = num_blocks
        self.torso = _PruneTorso(num_blocks, history_len, d_model, n_heads, n_layers, dim_feedforward)
        self.policy_head = _PolicyHeadAttn(d_model, n_heads=attn_heads)
        self.value_head = _ValueHeadAttn(d_model, d_hidden=dim_feedforward * 2, n_quantiles=8, n_heads=attn_heads)

    def forward(self, encoded_state, scalars):
        emb = self.torso(encoded_state, scalars)
        logits = self.policy_head(emb)
        quant = self.value_head(emb)
        value = _QuantileValueProj.risk_adjust(quant)
        return logits, value

    def fwd_train(self, states, scalars, pi, returns, lambda_H: float = 0.02):
        B = states.size(0)
        scalars = scalars.view(B, 1)
        emb = self.torso(states, scalars)
        logits = self.policy_head(emb)
        quant = self.value_head(emb)

        # policy loss (imitare π del MCTS)
        log_probs = F.log_softmax(logits, dim=-1)
        pol_loss = -(pi * log_probs).sum(dim=-1).mean()

        # entropia per esplorazione (solo come regolarizzazione, non backprop nel ritorno)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        ent_loss = entropy.detach()

        # value loss con quantile regression
        y = returns.view(B)
        val_loss = _QuantileValueProj.quantile_loss(quant, y)

        loss = pol_loss + val_loss - lambda_H * entropy
        return loss, pol_loss.detach(), val_loss.detach(), ent_loss

    @torch.no_grad()
    def fwd_infer(self, states, scalars, legal_mask=None, top_k=None):
        B = states.size(0)
        scalars = scalars.view(B, 1)
        logits, value = self.forward(states, scalars)

        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask == 0, -1e9)
        probs = F.softmax(logits, dim=-1)

        if top_k is not None:
            K = min(top_k, probs.size(-1))
            top_p, top_idx = torch.topk(probs, K, dim=-1)
        else:
            top_p, top_idx = probs, torch.arange(probs.size(-1), device=probs.device).unsqueeze(0).expand_as(probs)

        action_idx = torch.argmax(probs, dim=-1)
        return action_idx, probs, value, top_idx, top_p