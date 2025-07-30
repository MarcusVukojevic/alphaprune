import torch
from torch.utils.data import Dataset, ConcatDataset
from utils import tokens_to_action_tensor
from itertools import permutations, product
import random
import torch.nn.functional as F

class SyntheticDemoDataset(Dataset):
    def __init__(self, n_demos: int, R_limit: int,
                 device: str, include_terminal: bool = True):
        self.S, self.R = 4, R_limit
        self.device    = device
        self.inc_term  = include_terminal
        self.samples   = []
        for _ in range(n_demos):
            self._generate_demo()

    # -----------------------------------------------------------
    def _generate_demo(self):
        state   = torch.zeros(self.S, self.S, self.S, dtype=torch.int8)
        actions = []
        while len(actions) < self.R - 1:
            tok = torch.randint(0, 3, (3 * self.S,), dtype=torch.long)
            act = tokens_to_action_tensor(tok, self.S)
            if act.any():
                state += act
                actions.append(tok)

        future, last_tok = [], None
        for t, tok in enumerate(reversed(actions)):
            self._push_sample(state, future, tok, self.R - t - 1, -1.0)
            last_tok = tok                                   # memorizza ultimo
            state    -= tokens_to_action_tensor(tok, self.S)
            future.insert(0, tokens_to_action_tensor(tok, self.S))

        if self.inc_term and last_tok is not None:
            self._push_sample(state, future, last_tok, 0.0, 1.0)

    # -----------------------------------------------------------
    def _push_sample(self, state, future_frames, tok, scalar_val, reward_val):
        frames = [state.clone()] + future_frames
        frames += [torch.zeros_like(state)] * (self.R - len(frames))
        stack   = torch.stack(frames[: self.R])
        scalar  = torch.tensor([scalar_val], dtype=torch.float32)
        reward  = torch.tensor([reward_val], dtype=torch.float32)
        self.samples.append((stack, scalar, tok.clone(), reward))

    # -----------------------------------------------------------
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        st, sc, tok, rw = self.samples[i]
        return (st.to(self.device), sc.to(self.device),
                tok.to(self.device), rw.to(self.device))

def generate_base_changes(S=4, dtype=torch.int8):
    perms  = list(permutations(range(S)))
    signs  = list(product([-1, 1], repeat=S))
    for p in perms:
        P = torch.zeros(S, S, dtype=dtype)          # CPU
        P[range(S), list(p)] = 1
        for s in signs:
            D = torch.diag(torch.tensor(s, dtype=dtype))  # CPU
            yield D @ P



def get_strassen_factors(device:str):
    uu = torch.tensor([[1,0,0,1],
                       [0,0,1,1],
                       [1,0,0,0],
                       [0,0,0,1],
                       [1,1,0,0],
                       [-1,0,1,0],
                       [0,1,0,-1]], dtype=torch.int32, device=device)
    vv = torch.tensor([[1,0,0,1],
                       [1,0,0,0],
                       [0,1,0,-1],
                       [-1,0,1,0],
                       [0,0,0,1],
                       [1,1,0,0],
                       [0,0,1,1]], dtype=torch.int32, device=device)
    ww = torch.tensor([[1,0,0,1],
                       [0,0,1,-1],
                       [0,1,0,1],
                       [1,0,1,0],
                       [-1,1,0,0],
                       [0,0,0,1],
                       [1,0,0,0]], dtype=torch.int32, device=device)
    return uu, vv, ww


def encode_tokens(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    u,v,w: 1‑D torch int8 length S con coefficienti in {-1,0,1}.
    Ritorna: 1‑D torch int64 di lunghezza 3*S, con
      tok[0:S]     = u + 1,
      tok[S:2*S]   = v + 1,
      tok[2*S:3*S] = w + 1.
    In questo modo ogni posizione j ha valore 0 (se coeff=-1), 1 (se coeff=0), 2 (se coeff=+1).
    """
    S = u.numel()
    tok = torch.empty(3 * S, dtype=torch.int64)
    tok[0:S]       = (u.to(torch.int64) + 1)
    tok[S:2*S]     = (v.to(torch.int64) + 1)
    tok[2*S:3*S]   = (w.to(torch.int64) + 1)
    return tok


class StrassenAugDataset(Dataset):
    def __init__(self, n_demos: int, R_limit: int,
                 device: str, include_terminal: bool = True):
        self.S, self.R = 4, R_limit
        self.device    = device
        self.inc_term  = include_terminal
        self.samples   = []
        self._prepare(device, n_demos)

    # -----------------------------------------------------------
    def _prepare(self, device, n_demos):
        uu, vv, ww = get_strassen_factors(device)
        uu, vv, ww = uu.to(torch.int8), vv.to(torch.int8), ww.to(torch.int8)
        bases = list(generate_base_changes(self.S, torch.int8))

        for _ in range(n_demos):
            P, Q, Rm = (random.choice(bases) for _ in range(3))
            uu_t, vv_t, ww_t = (P @ uu.T).T, (Q @ vv.T).T, (Rm @ ww.T).T
            idx = torch.randperm(7, device=device)
            uu_t, vv_t, ww_t = uu_t[idx], vv_t[idx], ww_t[idx]
            actions = [encode_tokens(u, v, w) for u, v, w in zip(uu_t, vv_t, ww_t)]
            target  = sum(tokens_to_action_tensor(tok, self.S) for tok in actions)
            self._push_trajectory(actions, target)

    # -----------------------------------------------------------
    def _push_trajectory(self, actions, state):
        future, last_tok = [], None
        for t, tok in enumerate(reversed(actions)):
            self._push_sample(state, future, tok, self.R - t - 1, -1.0)
            last_tok = tok
            act   = tokens_to_action_tensor(tok, self.S)
            state -= act
            future.insert(0, act)

        if self.inc_term and last_tok is not None:
            self._push_sample(state, future, last_tok, 0.0, 1.0)

    # -----------------------------------------------------------
    def _push_sample(self, state, future_frames, tok, scalar_val, reward_val):
        frames = [state.clone()] + future_frames
        frames += [torch.zeros_like(state)] * (self.R - len(frames))
        stack   = torch.stack(frames[: self.R])
        scalar  = torch.tensor([scalar_val], dtype=torch.float32)
        reward  = torch.tensor([reward_val], dtype=torch.float32)
        self.samples.append((stack, scalar, tok.clone(), reward))

    # -----------------------------------------------------------
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        st, sc, tok, rw = self.samples[i]
        return (st.to(self.device), sc.to(self.device),
                tok.to(self.device), rw.to(self.device))