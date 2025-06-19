# parallel_alpha.py

import copy
import torch
import torch.multiprocessing as mp
from tqdm import trange
from parallel_actor import SelfPlayActor
from alpha          import AlphaZero

class AlphaZeroLearner(AlphaZero):
    def learn_async(self, sup_datasets, pretrain_epochs=1, n_actors=4, traj_queue_max=2000):

        # 0) Supervisd pre-training (grafico + checkpoint)
        super().learn_pretrain(sup_datasets, pretrain_epochs)
        print("finito pretraining...\ninizio iterazioni...")

        # 1) Inizializza una sola volta weights_queue/event
        mp.set_start_method("spawn", force=True)
        weights_event = mp.Event()
        weights_queue = mp.Queue(maxsize=1)

        # mettiamo i pesi iniziali in CPU
        init_sd_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}
        # primes the queue so that actors non blocchino al primo get()
        weights_queue.put(init_sd_cpu)
        weights_event.set()

        # 2) Loop sulle iterazioni
        for it in range(self.args["num_iterations"]):
            print(f"\n--- Iterazione {it} ---")

            # ricrea una nuova coda di traiettorie
            traj_queue = mp.Queue(maxsize=traj_queue_max)
            stop_event = mp.Event()

            # avvia gli attori
            actors = [
                SelfPlayActor(i, self.args, traj_queue, weights_event, weights_queue, stop_event)
                for i in range(n_actors)
            ]
            for a in actors:
                a.start()

            # 3) Raccogli self-play
            self.model.train()
            memory = []
            with trange(self.args["num_selfPlay_iterations"], desc=f"Self-play it {it}") as pbar:
                while len(memory) < self.args["num_selfPlay_iterations"]:
                    cpu_traj = traj_queue.get()
                    for st_cpu, sc_cpu, tok_cpu, rew_py in cpu_traj:
                        memory.append((
                            st_cpu.to(self.model.device, dtype=torch.float32),
                            sc_cpu.to(self.model.device, dtype=torch.float32),
                            tok_cpu.to(self.model.device),
                            torch.tensor(rew_py,
                                         dtype=torch.float32,
                                         device=self.model.device)
                        ))
                    pbar.update(len(cpu_traj))

            # 4) Ferma e joina gli attori **prima** del training
            stop_event.set()
            # svuota la coda per sbloccare eventuali put() pendenti
            while not traj_queue.empty():
                try:
                    traj_queue.get(False)
                except:
                    break

            for a in actors:
                a.join(timeout=5.0)

            # chiudi il feeder‐thread di traj_queue
            traj_queue.close()
            traj_queue.join_thread()

            print("Actor fermati e queue pulita — inizio training sulla memoria")

            # 5) Allena sul buffer raccolto
            print("Alleno sulle memorie...")
            for ep in range(self.args["num_epochs"]):
                print(f"  EP: {ep}")
                self.train(memory)
            print("Finito di allenare sulle memorie!")

            # 6) Aggiorna i pesi e sveglia gli actor per l’iterazione successiva
            init_sd_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}
            # replace old weights in queue
            # (rimuovo il vecchio se c’è, poi ne metto uno nuovo)
            if not weights_queue.empty():
                _ = weights_queue.get_nowait()
            weights_queue.put(init_sd_cpu)
            weights_event.set()

        # 7) Tutto fatto, chiudi weights_queue
        print("=== Tutte le iterazioni completate: chiudo weights_queue ===")
        weights_queue.close()
        weights_queue.join_thread()
        print("=== Training parallelo terminato e main ritorna al prompt ===")
