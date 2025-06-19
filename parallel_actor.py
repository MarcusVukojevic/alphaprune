import torch, torch.multiprocessing as mp
from games.prune_game import TensorGame
from model       import TensorModel
from alpha       import AlphaZero

class SelfPlayActor(mp.Process):
    def __init__(self, actor_id: int, args: dict, traj_queue: mp.Queue, weights_event, weights_queue: mp.Queue, stop_event):
        super().__init__(daemon=True)
        self.actor_id      = actor_id
        self.args          = args
        self.traj_queue    = traj_queue
        self.weights_event = weights_event
        self.weights_queue = weights_queue
        self.stop_event    = stop_event


    @torch.inference_mode()
    def run(self):
        torch.manual_seed(10 + self.actor_id)
        game = TensorGame(self.args)
        model = TensorModel(
            dim_3d=4, dim_t=8, dim_s=1, dim_c=16,
            n_steps=12, n_logits=3, n_samples=4,
            device=self.args['device']
        ).to(self.args['device'])
        self.weights_event.wait()
        init_sd = self.weights_queue.get()
        model.load_state_dict(init_sd, strict=False)
        self.weights_event.clear()
        model.eval()
        az = AlphaZero(model, optimizer=None, game=game, args=self.args)

        print(f"[Actor {self.actor_id}] avviato.")
        while not self.stop_event.is_set():
            # aggiorna pesi se disponibili
            if self.weights_event.is_set():
                new_sd = self.weights_queue.get()
                model.load_state_dict(new_sd, strict=False)
                model.to(self.args['device'])
                self.weights_event.clear()
                print(f"[Actor {self.actor_id}] ricevuto pesi nuovi.")

            # genera traiettoria
            trajs = az.selfPlay()

            # porta su CPU + format
            cpu_trajs = []
            for st, sc, tok, rew in trajs:
                cpu_trajs.append(( st.cpu().half(), sc.cpu().half(), tok.cpu(), float(rew)))

            # prova a fare put, ma se stop_event è stato settato, esci
            try:
                self.traj_queue.put(cpu_trajs, block=True, timeout=1.0)
            except:
                # timeout o queue full: controlla stop_event e continua
                continue

        print(f"[Actor {self.actor_id}] stop_event ricevuto, esco.")
