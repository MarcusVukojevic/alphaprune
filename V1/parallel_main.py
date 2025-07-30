import os, torch, argparse
from games.prune_game     import TensorGame
from model           import TensorModel
from dataset         import SyntheticDemoDataset, StrassenAugDataset
from parallel_alpha  import AlphaZeroLearner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Modalità test")
    return parser.parse_args()


def run(test=False):
    args = {
        "C": 2,
        "num_searches":           1 if test else 20,
        "num_iterations":         1 if test else 10, # quanti giri di selfplay-training
        "num_selfPlay_iterations":1 if test else 1000,
        "num_epochs":             1 if test else 10, # quante epoche nel traning selfplay-traning
        "num_epochs_pretrain":    1 if test else 100,
        "batch_size":             128 if test else 512,
        "models_path":            "models",
        "device":                 "mps" if test else "cuda",
        "T": 0,
        "R_limit": 8,
        "pretrained_path":        "models/mini_sage_fast_1.pt",
        "synth":  5 if test else 50000,
        "strass": 50 if test else 50000,
    }

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    game  = TensorGame(args)
    model = TensorModel(dim_3d=4, dim_t=8, dim_s=1, dim_c=16, n_steps=12, n_logits=3, n_samples=4, device=args["device"]).to(args["device"])

    # se esiste, carica checkpoint e salta pretrain
    ckpt = args.get("pretrained_path", "")
    if ckpt and os.path.isfile(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=args["device"]), strict=False)
        print(f"✓ modello caricato da {ckpt}")
        args["num_epochs_pretrain"] = 0

    optim   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    learner = AlphaZeroLearner(model, optim, game, args)

    learner.learn_async(
        sup_datasets=[
            SyntheticDemoDataset(args["synth"],  args["R_limit"], "cpu"),
            StrassenAugDataset(args["strass"], args["R_limit"], "cpu"),
        ],
        pretrain_epochs=args["num_epochs_pretrain"]
    )
    print("FINITO!")

if __name__ == "__main__":
    args = parse_args()
    run(test=args.test)
    run()
