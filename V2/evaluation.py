# ================== E V A L U A T I O N  ==================
import torch
from prune_game import PruneGame
from mcts import MCTS

def evaluate_current_model(model, args, save_plot: bool = True):
    """
    Crea un nuovo gioco, usa il modello corrente per decidere le mosse (via MCTS),
    e alla fine stampa PPL, sparsity e reward. Ritorna anche un dict con i risultati.
    """
    # 1) Nuovo environment/gioco “vergine”
    game_eval = PruneGame(args)
    state = game_eval.reset_game()

    # 2) Setup MCTS con gli stessi iperparametri
    mcts_eval = MCTS(game_eval, model, args)

    # 3) Metriche baseline
    ppl_baseline = game_eval.initial_ppl
    sparsity_start = 1.0 - state.float().mean().item()

    # 4) Loop decisionale (fino a done)
    while True:
        action = mcts_eval.search(state)          # gid da togglare
        state  = game_eval.do_action(action)      # aggiorna stato interno + ppl
        state = game_eval.state
        reward, done = game_eval.get_value_and_terminated(state, depth=game_eval.numero_mossa)
        if done:
            break

    state = game_eval.state
    # 5) Metriche finali
    ppl_final = game_eval.ppl
    sparsity_f = 1.0 - state.float().mean().item()
    n_steps = game_eval.numero_mossa

    if save_plot:
        game_eval.plot_scacchiera("eval_gate_state.png")
        game_eval.plot_reward_history("reward_metric_curve.png")


    print("\n======  E V A L U A T I O N  ======\n")
    print(f"PPL baseline : {ppl_baseline:.2f}")
    print(f"PPL finale   : {ppl_final:.2f}")
    print(f"Sparsity ini : {sparsity_start:.2%}")
    print(f"Sparsity fin : {sparsity_f:.2%}")
    print(f"Mosse fatte  : {n_steps}/{args['n_mosse_massimo']}")
    print(f"Reward finale: {reward:.4f}")
    if save_plot:
        print("🔖  plot stato porte in → eval_gate_state.png")

    return {
        "ppl_baseline": ppl_baseline,
        "ppl_final": ppl_final,
        "sparsity_start": sparsity_start,
        "sparsity_final": sparsity_f,
        "steps": n_steps,
        "reward": reward,
    }

