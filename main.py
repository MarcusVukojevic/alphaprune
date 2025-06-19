
import torch
from model import PruneModel
from alpha import AlphaZero
from games.prune_game import PruneGame

from utils import save_mask_png

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True


args = {
    "name_model"       : "distilgpt2",   # qualsiasi LLM HF
    "eightbit"         : False, 
    "name_dataset"     : "wikitext",
    "device"           : "mps",       # "cpu" o "mps" 
    "target_sparsity"  : 0.80,         # 50 %
    "ppl_tolerance_frac": 0.05,       # +0.5 % di ppl accettata
    "beta"             : 5.0,         # coeff. penalità Δppl
    "R_limit"          : 300,          # mosse max per episodio
    "num_searches"     : 64, #64
    "top_k"            : 64, #64
    "C"                : 1.5,
    "batch_size"       : 16, #16
    "num_iterations"   : 5,
    "num_selfPlay_iterations": 10,
    "num_epochs": 5,
    "beta": 3.0,           # peso KL nel reward
    "kl_threshold": 0.02   # τ per l’early-abort EvoPress
}

#!TODO: da guardare anche la ucb perché non mi quadra il 1- 



game = PruneGame(args)

n_blocks = game.initial_state.numel()

ppl_baseline = game.compute_perplexity(full_eval=True)
print(f"PPL baseline (distilgpt2 full): {ppl_baseline:.2f}\n\n")
model = PruneModel(num_blocks=n_blocks,history_len=args['R_limit'],d_model=128, n_heads=4, n_layers=2).to(args["device"])
optim  = torch.optim.Adam(model.parameters(), lr=1e-3)
alpha = AlphaZero(model, optim, game, args)

print("--> inizio ad imparare\n")
alpha.learn()

ppl_pruned = game.compute_perplexity()  
print(f"\nPPL modello potato: {ppl_pruned:.2f}")
