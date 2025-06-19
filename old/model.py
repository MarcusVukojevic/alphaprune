# prune_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PruneModel(nn.Module):
    """
    Un piccolo Policy/Value network per il PruneGame.
    - Input:  encoded_state  Tensor(B, T, N)
    - Output: policy_logits  Tensor(B, N, num_ops)
              value          Tensor(B,)
    - fwd_train: loss per policy (CE) e value (MSE)
    """

    def __init__(
        self,
        num_blocks: int,
        history_len: int,
        hidden_dim: int = 256,
        num_filters: list[int] = [64, 128, 256],
        num_ops: int = 3,
    ):
        super().__init__()
        self.num_blocks  = num_blocks
        self.history_len = history_len
        self.num_ops     = num_ops

        # Backbone conv1d gerarchica: (B, C=T, L=N) → (B, C_last, L')
        in_ch = history_len
        self.convs = nn.ModuleList()
        for out_ch in num_filters:
            self.convs.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, stride=2)
            )
            in_ch = out_ch

        # Calcola la dimensione L' dopo tutte le conv
        L = num_blocks
        for _ in num_filters:
            L = math.ceil(L / 2)
        self.flatten_dim = num_filters[-1] * L

        # Head policy: FC → logits (B, N×num_ops)
        self.policy_fc = nn.Linear(self.flatten_dim, num_blocks * num_ops)

        # Head value: FC → hidden → 1
        self.value_fc = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, encoded_state: torch.Tensor):
        """
        encoded_state: Tensor of shape (B, T, N)
        returns:
          p_logits (B, N, num_ops)
          value    (B,)
        """
        x = encoded_state                      # (B, T, N)
        # conv1d si aspetta (B, C, L)
        for conv in self.convs:
            x = F.relu(conv(x))                # dimezza L ad ogni layer

        B, C, L = x.shape
        x_flat = x.view(B, self.flatten_dim)   # (B, flatten_dim)

        # policy
        p = self.policy_fc(x_flat)             # (B, N*num_ops)
        p = p.view(B, self.num_blocks, self.num_ops)

        # value
        v = self.value_fc(x_flat).squeeze(-1)  # (B,)

        return p, v

    def fwd_train(
        self,
        states: torch.Tensor,
        scalars: torch.Tensor,    # passato ma non usato qui
        actions: torch.Tensor,    # (B, 2): [block_id, op]
        returns: torch.Tensor,    # (B,) o (B,1)
    ):
        """
        states:  (B, T, N)
        scalars: (B, 1)         (qui ignorati, ma puoi concatenarli se vuoi)
        actions: (B, 2) int64   block idx e op idx
        returns: (B, 1) float
        """
        B = states.size(0)

        # 1) forward
        logits, value = self.forward(states)  # logits (B,N,ops), value (B,)
        # 2) prepara per il CE
        #    flatten policy in (B, N*ops)
        logits_flat = logits.view(B, -1)      # (B, N*num_ops)

        # combini action tensor in indice unico: idx = block*num_ops + op
        block_idx = actions[:, 0].long()
        op_idx    = actions[:, 1].long()
        action_idx = block_idx * self.num_ops + op_idx  # (B,)

        # 3) policy loss
        pol_loss = F.cross_entropy(logits_flat, action_idx, reduction="mean")

        # 4) value loss
        returns = returns.view(-1).to(value.dtype)
        val_loss = F.mse_loss(value, returns, reduction="mean")

        return pol_loss, val_loss


    @torch.no_grad()
    def fwd_infer(self, states: torch.Tensor, scalars: torch.Tensor, top_k: int = 64):
        """
        Restituisce:
        • azioni candidate  LongTensor(B, K, 2)    [block, op]
        • prior_probs      Tensor  (B, K)
        • value            Tensor  (B,)
        """
        logits, value = self.forward(states)                 # (B,N,ops)

        B, N, num_ops = logits.shape
        logits_flat = logits.view(B, -1)                     # (B, N*ops)
        priors = torch.softmax(logits_flat, dim=-1)          # (B, N*ops)

        K = min(top_k, priors.size(-1))          # non chiedere più del possibile
        topk_vals, topk_idx = torch.topk(priors, k=K, dim=-1)

        # decodifica indici  idx = block*ops + op
        block_idx = (topk_idx // num_ops).long()
        op_idx    = (topk_idx  % num_ops).long()
        actions   = torch.stack([block_idx, op_idx], dim=-1)       # (B,K,2)

        return actions, topk_vals, value
    