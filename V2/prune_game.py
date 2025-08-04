

import torch
import torch.nn as nn
from patches import _patch_gpt2_block, _patch_llama_block
from utils import load_model, build_calib_dataset
import math
from collections import deque


class PruneGame():
    # funzione per applicare la patch al modello
    @staticmethod
    def _ensure_patched(model: nn.Module):
        for blk in model.modules():
            if hasattr(blk, "attn") and hasattr(blk, "mlp") and hasattr(blk, "ln_1"):
                _patch_gpt2_block(blk)
            elif hasattr(blk, "self_attn") and hasattr(blk, "mlp") and hasattr(blk, "input_layernorm"):
                _patch_llama_block(blk)
        return model

    # funzione che mi restituisce il tensore stato del gioco e una lista con i ResidualGates
    @staticmethod
    def _collect_gates(model):
        ptrs = []
        for blk in model.modules():
            if hasattr(blk, "g_mha") and hasattr(blk, "g_ffn"):
                ptrs.extend([blk.g_mha, blk.g_ffn])   # sempre due per layer
        if not ptrs:
            raise ValueError("Patch failed – no gates found.")
        state = torch.ones(len(ptrs), dtype=torch.int8)
        return state, ptrs

    def __init__(self, args):
        
        # Varibili iniziali e utili
        self.model_name = args["model_name"]
        self.device = args["device"]
        self.eight_bit = False
        self.target_sparsity = args["target_sparsity"]
        self.numero_mossa = 0 # a che mossa siamo durante il gioco
        self.limite_mosse = args["n_mosse_massimo"]
        self.nome_dataset = args["name_dataset"]
        self.eval_mode = args["eval_mode"]
        self.tol_frac = 0.1
        self.args = args

        
        # modello e tokenizer
        self.model_victim = PruneGame._ensure_patched(load_model(self.model_name, device=self.device, eightbit=self.eight_bit))
        self.model_victim.config.use_cache = False # safe init
        self.tokenizer = self.model_victim.tokenizer
        
        # inizializzo la board
        self.state, self.gates = PruneGame._collect_gates(self.model_victim)
        self.state = self.state.to(self.device)
        self.initial_state = self.state.clone() # stato iniziale fisso

        # TODO: da stare attenti ad un possibile POF sulla lunghezza delle sequenze
        self.dataset = build_calib_dataset(self.nome_dataset, self.tokenizer, split="validation", nsamples=10, seq_len=128)

        self.ppl = self.compute_ppl()
        self.initial_ppl = self.ppl
        print("--> ppl iniziale: ", self.ppl)

        if self.eval_mode == "mse":
            self._cache_teacher_full()
            self.scale_mse = (self.teacher_full ** 2).mean().item() 
        self.mse = 0

        # per debug e logging
        self.history = deque(maxlen=self.limite_mosse - 1)
        self.history.appendleft(self.initial_state)
        self.reward_history = []
        self.ppl_history = []
        self.mse_history = []

    def _cache_teacher_full(self, batch_size=64, dtype=torch.float16):
        self.model_victim.eval()
        ref_lp, inputs = [], []
        with torch.no_grad():
            for i in range(0, len(self.dataset), batch_size):
                j   = min(i + batch_size, len(self.dataset))
                inp = torch.stack(self.dataset[i:j], dim=0).long().to(self.device)
                lp  = torch.log_softmax(self.model_victim(inp).logits, dim=-1)  # (B,L,V)
                ref_lp.append(lp.to(dtype).cpu())
                inputs.append(inp.cpu())
        self.teacher_full = torch.cat(ref_lp, dim=0)      # (N,L,V) fp16 CPU
        self.calib_inputs_cpu = torch.cat(inputs, dim=0)

    @torch.no_grad()
    def compute_mse(self, batch_size=64) -> float:
        self.model_victim.eval()
        tot, count = 0.0, 0
        N = self.teacher_full.size(0)
        for i in range(0, N, batch_size):
            j   = min(i + batch_size, N)
            inp = self.calib_inputs_cpu[i:j].to(self.device)
            lp  = torch.log_softmax(self.model_victim(inp).logits, dim=-1) # (B,L,V)
            ref = self.teacher_full[i:j].to(self.device, dtype=lp.dtype)
            mse = torch.mean((lp - ref) ** 2).item()
            bsz = (j - i)
            tot += mse * bsz
            count += bsz
        return tot / max(count, 1)

    def reset_game(self):
        # resetto lo stato
        self.state.copy_(self.initial_state)
        # resetto i gates
        for g in self.gates:
            g.alpha.data.fill_(1.0)
        
        self.ppl = self.initial_ppl
        self.initial_mse = 0
        self.numero_mossa = 0
        self.history = []
        self.history = deque(maxlen=self.limite_mosse - 1)
        self.history.appendleft(self.initial_state)
        #self.reward_history = []
        #self.ppl_history = []
        #self.mse_history = []

        return self.state
    
    # dato uno stato mi controlla se ho vinto
    # vinco se ho raggiunto:
    #   1) la sparsity che volevo
    #   2) minor ppl possibile
    def controllo_vittoria(self, state):
        sparsity = 1.0 - state.float().mean().item()
        ok_s = sparsity >= self.target_sparsity
        if self.eval_mode == "ppl":
            ppl_target = self.initial_ppl * (1.0 + self.tol_frac)
            ok_m = self.ppl <= ppl_target
        else:  # mse
            mse_target = self.scale_mse * self.tol_frac
            ok_m = self.mse <= mse_target
        return ok_s and ok_m

    
    def get_value_and_terminated(self, state, depth=None, register=False):
        """
        Ritorna (reward_finale, done) SOLO se episodio terminato.
        Altrimenti (0.0, False). Calcola PPL realmente ogni step come vuoi tu.
        """
        # ----- metriche attuali -----
        sparsity = 1.0 - state.float().mean().item()
        # ----- condizioni -----
        hit_s = sparsity >= self.target_sparsity
        mse_penalty = 0.0   
        if self.eval_mode == "ppl":
            ppl_now  = self.compute_ppl()
            # Definisci una soglia PPL: baseline * (1 + tol_frac)
            ppl_target = self.initial_ppl * (1.0 + self.tol_frac)
            hit_p = ppl_now <= ppl_target
        elif self.eval_mode == "mse":
            mse_now  = self.compute_mse()
            target   = self.scale_mse * self.tol_frac     # ← soglia dinamica
            mse_penalty = max(0.0, min(1.0, mse_now / target))  # 0‑1
            hit_p    = mse_now <= target
        steps = self.numero_mossa if depth is None else depth
        limit = steps >= self.limite_mosse
        win   = hit_s and hit_p
        done  = win or limit

        if not done:
            return 0.0, False

        # ----- normalizzazioni -----
        # Sparsity norm
        s_norm = (sparsity - self.target_sparsity) / (1.0 - self.target_sparsity + 1e-8)
        s_norm = float(max(0.0, min(1.0, s_norm)))

        # PPL penalty
        if self.eval_mode == "ppl":
            rel_diff = (ppl_now - self.initial_ppl) / (self.initial_ppl + 1e-8)
            ppl_penalty = rel_diff / self.tol_frac   # quanto oltre la tolleranza
            ppl_penalty = float(max(0.0, min(1.0, ppl_penalty)))
        #elif self.eval_mode == "mse":
        #    rel_diff = (mse_now - self.initial_mse) / (1e-8)
        #    mse_penalty = rel_diff / tol_frac   # quanto oltre la tolleranza
        #    mse_penalty = float(max(0.0, min(1.0, mse_penalty)))


        # Efficienza mosse
        eff = 1.0 - steps / (self.limite_mosse + 1e-8)
        eff = float(max(0.0, min(1.0, eff)))

        # ----- combinazione -----
        if win:
            w_s, w_p, w_e = 0.5, 0.4, 0.1
            if self.eval_mode == "ppl":
                reward = w_s * s_norm + w_p * (1.0 - ppl_penalty) + w_e * eff
            elif self.eval_mode == "mse":
                reward = w_s * s_norm + w_p * (1.0 - mse_penalty) + w_e * eff
        elif hit_s and not hit_p:
            # casi intermedi: hai fatto la sparsity ma ppl peggiorata troppo
            if self.eval_mode == "ppl":
                reward = 0.3 * s_norm - 0.7 * ppl_penalty
            elif self.eval_mode == "mse":
                reward = 0.3 * s_norm - 0.7 * mse_penalty
        else:
            # fail: finito mosse senza arrivare
            miss_s = 1.0 - s_norm
            if self.eval_mode == "ppl":
                miss_p = ppl_penalty
            elif self.eval_mode == "mse":
                miss_p = mse_penalty
            reward = -0.5 * (miss_s + miss_p)

        # Clippa
        reward = float(max(-1.0, min(1.0, reward)))

        if register:
            self.reward_history.append(reward)
        return reward, True


    # funzione che dato uno stato restituisce il tensore da passare al modello
    #! TODO occhio: devo capire dove va a finire questa azione perché :
    #       1) se la uso in mcts io non ho storia in teoria
    #       2) se la uso quando faccio le azioni si, però allora non ho bisogno di passargli uno stato perché self.state è l'ultimo stato
    def get_encoded_state(self, state: torch.Tensor):

        T, N = self.limite_mosse, state.numel()
        enc  = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        # stato corrente
        enc[0] = state.float()
        # il resto è la nostra storia
        for t, past_state in enumerate(self.history, start=1):
            if t >= T:
                break
            enc[t] = past_state.float()

        #!TODO da tenere d'occhio questa cosa perché è una matrice
        return enc
    

    def get_scalar(self):
        return torch.tensor([self.limite_mosse - self.numero_mossa], dtype=torch.float32, device=self.device)

    # L'unica azione disponibile
    def toggle_gate(self, gid:int):
        
        # prendo il gate di riferimento
        gate = self.gates[gid]
        nuovo_valore = 1 - int(self.state[gid].item())
        gate.alpha.data.fill_(float(nuovo_valore))
        self.state[gid] = nuovo_valore


    def do_action(self, gid:int):
        self.toggle_gate(gid)
        self.numero_mossa += 1
        self.history.appendleft(self.state.clone())
        if self.eval_mode == "ppl":
            self.ppl = self.compute_ppl()  
            self.ppl_history.append(self.ppl)  
        elif self.eval_mode == "mse":
            self.mse = self.compute_mse()
            self.mse_history.append(self.mse)

    # sarebbe un toggle, senza veramente toccare la board -> utile per mcts
    def pensa_azione(self, state: torch.Tensor, action: int):
        # state in qeusto caso sarebbe la nostra board, non self.state
        nuovo_valore = 1 - state[action].item()
        nuovo_stato = state.clone()
        nuovo_stato[action] = nuovo_valore
        return nuovo_stato

    def _cache_teacher_full(self, batch_size=64, dtype=torch.float16):
        self.model_victim.eval()
        ref_lp, inputs = [], []
        with torch.no_grad():
            for i in range(0, len(self.dataset), batch_size):
                j   = min(i + batch_size, len(self.dataset))
                inp = torch.stack(self.dataset[i:j], dim=0).long().to(self.device)
                lp  = torch.log_softmax(self.model_victim(inp).logits, dim=-1)  # (B,L,V)
                ref_lp.append(lp.to(dtype).cpu())
                inputs.append(inp.cpu())
        self.teacher_full = torch.cat(ref_lp, dim=0)      # (N,L,V) fp16 CPU
        self.calib_inputs_cpu = torch.cat(inputs, dim=0)

    @torch.no_grad()
    def compute_mse_full_logits(self, batch_size=64) -> float:
        self.model_victim.eval()
        tot, count = 0.0, 0
        N = self.teacher_full.size(0)
        for i in range(0, N, batch_size):
            j   = min(i + batch_size, N)
            inp = self.calib_inputs_cpu[i:j].to(self.device)
            lp  = torch.log_softmax(self.model_victim(inp).logits, dim=-1) # (B,L,V)
            ref = self.teacher_full[i:j].to(self.device, dtype=lp.dtype)
            mse = torch.mean((lp - ref) ** 2).item()
            bsz = (j - i)
            tot += mse * bsz
            count += bsz
        return tot / max(count, 1)


    @torch.no_grad()
    def compute_ppl(self, batch_size: int = 64) -> float:
        self.model_victim.eval()

        total_nll, total_tok = 0.0, 0
        loader = self.dataset
        dev = self.device
        for i in range(0, len(loader), batch_size):
            j = min(i + batch_size, len(loader))
            inp = torch.stack(loader[i:j], dim=0).long().to(dev)
            outputs = self.model_victim(inp, labels=inp, use_cache=False)
            loss    = outputs.loss.float()           # già shiftata
            total_nll += loss.item() * (inp.size(1)-1)
            total_tok += (inp.size(1)-1)

        ppl = math.exp(total_nll / total_tok)
        return ppl
    
    @torch.no_grad()
    def compute_kl(self):
        pass

    @torch.no_grad()
    def plot_scacchiera(self, fname="gate_state.png"):
        import matplotlib.pyplot as plt
        n_layers = self.state.numel() // 2
        mat = self.state.view(n_layers, 2).cpu().numpy()
        fig, ax = plt.subplots(figsize=(4, n_layers * 0.35 + 1.5))
        im = ax.imshow(mat, cmap=plt.cm.get_cmap("Greys", 2), vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["MHA", "FFN"])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_title("Gate state (1=ON, 0=OFF)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"🔖  plot salvato in → {fname}")


    @torch.no_grad()
    def plot_reward_history(self, fname: str = "reward_history.png"):
        """
        Disegna la curva del reward step-by-step usando self.reward_history.
        """
        import matplotlib.pyplot as plt

        if not self.reward_history:
            print("[warn] reward_history è vuota – grafico non generato")
            return

        steps = range(1, len(self.reward_history) + 1)
        plt.figure(figsize=(8, 4))
        plt.plot(steps, self.reward_history, lw=2, marker="o", color="tab:blue")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Reward ad ogni step dell'episodio")
        plt.grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"📈  curva reward salvata in → {fname}")