# main.py
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
        # --- Modello vittima & dataset HF ---
        "model_name": "meta-llama/Llama-2-7b-hf",  # es: "distilgpt2", "gpt2", "Qwen/Qwen2-0.5B"
        "name_dataset": "wikitext",                # "wikitext" => wikitext-2-raw-v1 (val)
        "device": device,

        # --- Target pruning ---
        "target_sparsity": 0.50,       # frazione di gate OFF desiderata
        "n_mosse_massimo": 20,         # orizzonte T
        "eval_mode": "ppl",            # "ppl" | "kl" | "mse"

        # --- AlphaZero loop ---
        "num_episodes": 50,            # quante iterazioni globali
        "num_self_iteration": 20,      # quanti episodi di self-play per iter
        "num_epochs": 5,               # passi di training per iter
        "batch_size": 32,
        "grad_clip": 1.0,
        "entropy_bonus": 0.02,
        "lr": 2e-4,
        "replay_maxlen": 10000,        # cap del buffer

        # --- MCTS ---
        "C": 2.0,
        "num_searches": 32,            # >7 per spazio azioni ampio
        "top_k": 16,                   # abilita dopo warmup se vuoi
        "gamma": 0.97,                 # preferisce piani corti
        "depth_penalty": 0.01,         # piccolo costo per step
        "root_noise_eps": 0.25,        # Dirichlet alla root
        "root_dirichlet_alpha": 0.3,

        # --- Tolleranze / shaping PPL ---
        "ppl_tol_frac": 0.10,          # +10% rispetto alla baseline ok
        "ppl_alpha": 0.05,             # morbidezza penalità PPL
        "s_beta": 0.05,                # picco su target sparsity
        "eps_sparsity_bonus": 0.2,     # bonus su vicinanza al target

        # --- Parallel / Esperimenti ---
        "parallel": True,              # << flag: True = self-play parallelo, False = sequenziale
        "parallel_workers": 4,         # num thread per il self-play (usato solo se parallel=True)
        "exp_id": 1,                   # numero esperimento: usato come suffisso file (opzionale)
    }

    print(args)
    # Env di pruning (patcha il modello, prepara dataset/calib, ecc.)
    game = PruneGame(args)

    # Policy+Value per controllare i gate
    num_blocks = game.initial_state.numel()   # N gates = 2*#layers
    history_len = args["n_mosse_massimo"]     # T
    model = PruneModel(num_blocks, history_len).to(args["device"])

    # AlphaZero loop
    az = AlphaZero(model, game, args)
    print("\n--> Inizio training AlphaPrune\n")
    az.learn()

    # Suffix esperimento
    exp = args.get("exp_id", None)
    suffix = f"_{exp}" if exp is not None else ""

    # Plot training reward e valutazione finale con plot stato/curve
    game.plot_reward_history(f"4_reward_curve_training{suffix}.png")
    print("\n--> Valutazione finale con MCTS sul modello corrente\n")
    args["plot_suffix"] = suffix  # se evaluation.py usa il suffisso
    _ = evaluate_current_model(model, args, save_plot=True)
