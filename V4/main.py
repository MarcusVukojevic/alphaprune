import torch
from alphazero import AlphaZero
from model import PruneModel
from prune_game import PruneGame
from evaluation import evaluate_current_model

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

if __name__ == "__main__":
    device = pick_device()
    args = {
        "model_name"   : "meta-llama/Llama-2-7b-hf",  # es: "distilgpt2"
        "name_dataset" : "wikitext",                  
        "device"       : device,
        "calib_nsamples": 5,
        "calib_seq_len" : 50,

        # --- Target pruning ---
        "target_sparsity"  : 0.37,       # frazione di gate OFF desiderata
        "n_mosse_massimo"  : 20,         # orizzonte T
        "eval_mode"        : "ppl",      # "ppl" | "kl" | "mse"

        # --- AlphaZero loop ---
        "num_episodes"      : 50,        # iterazioni globali
        "num_self_iteration": 20,        # episodi di self-play per iter
        "num_epochs"        : 5,         # passi di training per iter
        "batch_size"        : 32,
        "grad_clip"         : 1.0,
        "entropy_bonus"     : 0.02,
        "lr"                : 2e-4,
        "replay_maxlen"     : 10000,     # cap del buffer

        # --- MCTS ---
        "C"             : 2.0,
        "num_searches"  : 32,
        "top_k"         : 8,
        "gamma"         : 0.97,
        "depth_penalty" : 0.01,
        "root_noise_eps": 0.25,
        "root_dirichlet_alpha": 0.3,

        # --- Tolleranze / shaping PPL ---
        "ppl_tol_frac"  : 0.10,
        "ppl_alpha"     : 0.05,
        "s_beta"        : 0.05,
        "eps_sparsity_bonus": 0.2,

        # --- Parallel (stessa GPU) ---
        "parallel"          : False,   # True -> usa parallel_learn(); False -> learn() sequenziale
        "parallel_workers"  : 2,       # quante istanze di PruneGame sulla stessa device
        "metric_cache_cap"  : 20000,   # capacità LRU condivisa
    }

    print(args)
    # Env di pruning (patcha il modello, prepara dataset/calib, ecc.)
    game = PruneGame(args)

    # Policy+Value per controllare i gate
    num_blocks   = game.initial_state.numel()   # N gates = 2*#layers
    history_len  = args["n_mosse_massimo"]      # T
    model = PruneModel(num_blocks, history_len).to(args["device"])

    # AlphaZero loop
    az = AlphaZero(model, game, args)
    print("\n--> Inizio training AlphaPrune\n")
    if args.get("parallel", False):
        az.parallel_learn()
    else:
        az.learn()

    # Plot training reward e valutazione finale con plot stato/curve
    game.plot_reward_history("4_reward_curve_training.png")
    print("\n--> Valutazione finale con MCTS sul modello corrente\n")
    _ = evaluate_current_model(model, args, save_plot=True)