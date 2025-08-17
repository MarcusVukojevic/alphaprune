# evaluation.py
from prune_game import PruneGame
from mcts import MCTS


def evaluate_current_model(model, args, save_plot: bool = True):
    # Forza eval in PPL, lasciando inalterato tutto il resto
    args_eval = dict(args)
    args_eval["eval_mode"] = "ppl"

    # crea un nuovo env (ricarica modello vittima patchato, dataset calib, ecc.)
    game_eval = PruneGame(args_eval)
    _ = game_eval.reset_game()

    # MCTS eval con stessi parametri (num_searches, ecc.)
    mcts_eval = MCTS(game_eval, model, args_eval)

    ppl_baseline = game_eval.initial_ppl
    state = game_eval.state
    sparsity_start = 1.0 - state.float().mean().item()

    while True:
        state = game_eval.state.clone()
        action = mcts_eval.search(state)
        state = game_eval.do_action(action)
        state = game_eval.state
        reward, done = game_eval.get_value_and_terminated(
            state, depth=game_eval.numero_mossa, register=True
        )
        if done:
            break

    state = game_eval.state
    ppl_final = game_eval.compute_ppl()  # usa lo shim, coerente con calib corrente
    sparsity_f = 1.0 - state.float().mean().item()
    n_steps = game_eval.numero_mossa

    # suffix esperimento se presente
    suffix = args.get("plot_suffix", "")
    if save_plot:
        game_eval.plot_scacchiera(f"4_eval_gate_state{suffix}.png")
        game_eval.plot_reward_history(f"4_reward_metric_curve{suffix}.png")

    print("\n======  E V A L U A T I O N  ======\n")
    print(f"PPL baseline : {ppl_baseline:.2f}")
    print(f"PPL finale   : {ppl_final:.2f}")
    print(f"Sparsity ini : {sparsity_start:.2%}")
    print(f"Sparsity fin : {sparsity_f:.2%}")
    print(f"Mosse fatte  : {n_steps}/{args['n_mosse_massimo']}")
    print(f"Reward finale: {reward:.4f}")
    if save_plot:
        print(f"🔖  plot stato porte in → 4_eval_gate_state{suffix}.png")

    return {
        "ppl_baseline": ppl_baseline,
        "ppl_final": ppl_final,
        "sparsity_start": sparsity_start,
        "sparsity_final": sparsity_f,
        "steps": n_steps,
        "reward": reward,
    }
