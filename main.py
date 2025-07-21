
import torch
#from model import PruneModel
from new_model import PruneModel
from alpha import AlphaZero
from games.prune_game import PruneGame

from utils import save_mask_png

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args_big = {
    "name_model"       : "distilgpt2",   # qualsiasi LLM HF
    "eightbit"         : False, 
    "name_dataset"     : "wikitext",
    "device"           : "cuda",       # "cpu" o "mps" 
    "target_sparsity"  : 0.50,         # 50 %
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
    "name_model"       : "meta-llama/Llama-2-7b-hf",  #  "distilgpt2",# qualsiasi LLM HF
    "eightbit"         : False, 
    "name_dataset"     : "wikitext",
    "device"           : "cuda",       # "cpu" o "mps" 
    "target_sparsity"  : 0.30,         # 50 %
    "R_limit"          : 60,          # mosse max per episodio
    "num_searches"     : 64, #64
    "top_k"            : 64, #64
    "C"                : 3,#1.5,
    "batch_size"       : 48, #16
    "num_iterations"   : 50,
    "num_selfPlay_iterations": 50,
    "num_epochs": 7,
    "beta": 2,           # peso KL nel reward
    "kl_threshold": .2,   # τ per l’early-abort EvoPress e checkwin
    "root_dir_eps": 0.7,
    "root_dir_alpha": 0.3,
    "lr": 2e-4,
    "entropy_bonus": 0.02,
    "grad_clip": 1.0,
    "mcts_batch_size": 3, 
}

#!TODO: da guardare anche la ucb perché non mi quadra il 1- 

args = args_mini


print("Inizio traning con: ", str(args))

game = PruneGame(args)

n_blocks = game.initial_state.numel()

ppl_baseline = game.compute_perplexity(full_eval=True)
print(f"PPL baseline: {ppl_baseline:.2f}\n\n")

#print(game.perform_action(torch.Tensor([3,0])))
#print(game.perform_action(torch.Tensor([4,0])))
#print(game.perform_action(torch.Tensor([5,0])))
#print(game.perform_action(torch.Tensor([8,0])))
#print(game.perform_action(torch.Tensor([9,0])))
#print(game.perform_action(torch.Tensor([11,0])))
#print(game.kl_div)
#print(game.reward)
#
#ppl_pruned = game.compute_perplexity()  
#final_sparsity = 1.0 - game.state.float().mean().item()
#print(f"\nPPL modello potato: {ppl_pruned:.2f}")
#print(f"Sparsity finale   : {final_sparsity:.2%}") 
#game.plot_gate_state("gugu.png")
#exit()


model = PruneModel(num_blocks=n_blocks, history_len=args['R_limit'], d_model=128, n_heads=8, n_layers=5).to(args["device"])
if torch.version.cuda and torch.cuda.is_available():
    model = torch.compile(model, mode="reduce-overhead")
print("modellio compilato e pront! ")


optim = torch.optim.AdamW(model.parameters(), lr=args['lr'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args["num_epochs"])


alpha = AlphaZero(model, optim, game, scheduler, args)

print("--> inizio ad imparare\n")
alpha.learn()

ppl_pruned = game.compute_perplexity()  
final_sparsity = 1.0 - game.state.float().mean().item()
print(f"\nPPL modello potato: {ppl_pruned:.2f}")
print(f"Sparsity finale   : {final_sparsity:.2%}") 

game.plot_gate_state("final_gate_state.png")




# -------------------------------------------------------------------
#  E V A L U A T I O N   (stesso script, dopo il training)
# -------------------------------------------------------------------
print("\n======  E V A L U A T I O N  ======\n")
from mcts import MCTS

# 1) nuova scacchiera
eval_game = PruneGame(args)                       # stato tutto ON

# 2) nuovo modello + pesi addestrati
eval_model = model
eval_model.eval()                                                 # no grad

# 3) MCTS “greedy”: niente Dirichlet e poche simulazioni veloci
eval_args = args
mcts_eval = MCTS(eval_game, eval_args, eval_model)

state = eval_game.get_initial_state()
while True:
    action = mcts_eval.search(state)              # azione migliore
    print(action)
    state  = eval_game.perform_action(action)     # applicala
    _, done = eval_game.get_value_and_terminated(state)
    if done:
        break

# ---- metriche finali ---------------------------------------------
ppl_pruned = eval_game.compute_perplexity()                 # dopo potatura
spars_f    = 1.0 - state.float().mean().item()

print(f"[EVAL] PPL baseline : {ppl_baseline:.2f}")
print(f"[EVAL] PPL pruned   : {ppl_pruned:.2f}")
print(f"[EVAL] Sparsity     : {spars_f:.2%}")

eval_game.plot_gate_state("eval_gate_state.png")
print("✅  evaluation completa – vedi eval_gate_state.png")
