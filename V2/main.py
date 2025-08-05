from AlphaZero import AlphaZero
from model import PruneModel
from prune_game import PruneGame
from evaluation import evaluate_current_model

args = {
    # import relativi a modelli e dati
    "model_name" : "distilgpt2",
    "name_dataset" : "wikitext",
    "device": "cuda",

    # target di sparsity
    "target_sparsity" : 0.2,

    # iper params per Alphazero
    "num_episodes": 100,
    "num_self_iteration" : 20,
    "num_epochs" : 5, # --> numero epoche per il modello
    "n_mosse_massimo" : 20,
    "batch_size": 32,
    "grad_clip": 1.0,
    "entropy_bonus": 0.02,
    "lr": 2e-4,

    # iper params per MCTS
    "C": 2,
    "num_searches" : 25,
    "top_k": 3,

    # che modalità di evaluation vogliamo? mse | ppl
    "eval_mode" : "ppl"
}

gioco = PruneGame(args)
stato_iniziale = gioco.initial_state

model = PruneModel(gioco.initial_state.numel(), args["n_mosse_massimo"]).to(args["device"])
alphazero = AlphaZero(model, gioco, args)

print("--> inizio ad imparare")
alphazero.learn()


gioco.plot_reward_history("reward_curve_traninig.png")
results = evaluate_current_model(model, args)
