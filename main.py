
import torch
from model import PruneModel
from alpha import AlphaZero
from games.prune_game import PruneGame

from utils import save_mask_png

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True


args_big = {
    "name_model"       : "distilgpt2",   # qualsiasi LLM HF
    "eightbit"         : False, 
    "name_dataset"     : "wikitext",
    "device"           : "cuda",       # "cpu" o "mps" 
    "target_sparsity"  : 0.90,         # 50 %
    "ppl_tolerance_frac": 0.05,       # +0.5 % di ppl accettata
    "beta"             : 5.0,         # coeff. penalità Δppl
    "R_limit"          : 30,          # mosse max per episodio
    "num_searches"     : 64, #64
    "top_k"            : 32, #64
    "C"                : 1.5,
    "batch_size"       : 32, #16
    "num_iterations"   : 50,
    "num_selfPlay_iterations": 100,
    "num_epochs": 50,
    "beta": 1.0,           # peso KL nel reward
    "kl_threshold": 0.1   # τ per l’early-abort EvoPress
}


args_mini = {
    "name_model"       : "distilgpt2",#"meta-llama/Llama-2-7b-hf",   # qualsiasi LLM HF
    "eightbit"         : False, 
    "name_dataset"     : "wikitext",
    "device"           : "mps",       # "cpu" o "mps" 
    "target_sparsity"  : 0.20,         # 50 %
    "ppl_tolerance_frac": 0.1,       # +0.5 % di ppl accettata
    "beta"             : 5.0,         # coeff. penalità Δppl
    "R_limit"          : 30,          # mosse max per episodio
    "num_searches"     : 10, #64
    "top_k"            : 10, #64
    "C"                : 1.5,
    "batch_size"       : 12, #16
    "num_iterations"   : 50,
    "num_selfPlay_iterations": 20,
    "num_epochs": 2,
    "beta": 0.5,           # peso KL nel reward
    "kl_threshold": 0.5,   # τ per l’early-abort EvoPress
    "root_dir_eps": 0.3,
    "root_dir_alpha": 0.3,
}

#!TODO: da guardare anche la ucb perché non mi quadra il 1- 

args = args_mini


print("Inizio traning con: ", str(args))

game = PruneGame(args)

n_blocks = game.initial_state.numel()

ppl_baseline = game.compute_perplexity(full_eval=True)
print(f"PPL baseline (distilgpt2 full): {ppl_baseline:.2f}\n\n")
model = PruneModel(num_blocks=n_blocks, history_len=args['R_limit'],d_model=128, n_heads=8, n_layers=5).to(args["device"])
optim  = torch.optim.Adam(model.parameters(), lr=1e-4)
alpha = AlphaZero(model, optim, game, args)

print("--> inizio ad imparare\n")
alpha.learn()

ppl_pruned = game.compute_perplexity()  
print(f"\nPPL modello potato: {ppl_pruned:.2f}")

game.plot_gate_state("final_gate_state.png")