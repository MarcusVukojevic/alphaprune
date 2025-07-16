import torch
from node import Node

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        self.Cpuct = args.get("C", 1.0)
        # NUOVO: Dimensione del batch per le inferenze durante la ricerca MCTS
        self.mcts_batch_size = args.get("mcts_batch_size", 8)

    @torch.no_grad()
    def search(self, root_state):
        # 1. Preparazione Iniziale
        root = Node(state=root_state, parent=None, action_taken=None, prior=1.0)
        self.last_root = root
        
        # Espandi la radice per prima cosa, da sola, per avere le policy iniziali e applicare il rumore
        # (Questo conta come la prima simulazione)
        self._expand_batch_and_propagate([root])
        sims_done = 1

        # 2. Ciclo di Ricerca Principale (finché non esauriamo il budget di simulazioni)
        while sims_done < self.args["num_searches"]:
            leaves_to_expand = []
            
            # FASE A: Raccolta di un batch di nodi "foglia"
            for _ in range(self.mcts_batch_size):
                node = root
                
                # Selezione: scendi lungo l'albero fino a un nodo foglia
                while node.children:
                    node = node.select(self.Cpuct)

                # Controlla se il nodo è terminale (fine partita)
                reward, done = self.game.get_value_and_terminated(node.state)
                if done:
                    # Se è terminale, propaga il suo valore reale e non espanderlo.
                    # Questo è cruciale per evitare blocchi.
                    node.backpropagate(reward)
                    sims_done += 1
                else:
                    # Se non è terminale, è una foglia da espandere.
                    leaves_to_expand.append(node)
                
                # Se abbiamo esaurito il budget durante la raccolta, esci
                if sims_done + len(leaves_to_expand) >= self.args["num_searches"]:
                    break
            
            # Se non abbiamo raccolto foglie (es. tutti i percorsi portano a nodi terminali), usciamo
            if not leaves_to_expand:
                break

            # FASE B & C: Espansione del batch e backpropagation
            self._expand_batch_and_propagate(leaves_to_expand)
            sims_done += len(leaves_to_expand)

        # 3. Scelta finale dell'azione
        best_child = max(root.children, key=lambda c: c.visit_count)
        return best_child.action_taken


    def _expand_batch_and_propagate(self, nodes):
        """
        Funzione helper che prende una lista di nodi, esegue l'inferenza in batch,
        espande ogni nodo con la sua policy e propaga il suo valore.
        """
        if not nodes:
            return

        # Prepara l'input per la rete in un unico batch
        dev = next(self.model.parameters()).device
        
        # La chiamata a get_encoded_state usa la storia del gioco principale, che è una semplificazione
        # ma è consistente con il design attuale del tuo codice.
        encoded_states = torch.stack([self.game.get_encoded_state(n.state) for n in nodes])
        scalars = torch.stack([self.game.get_scalar() for n in nodes]) # Scalari sono uguali per tutti i nodi di una ricerca
        
        # Esegui una sola inferenza per l'intero batch
        action_probs, priors_batch, values_batch = self.model.fwd_infer(
            encoded_states, 
            scalars, 
            top_k=self.args.get("top_k", 32)
        )

        # Per ogni nodo nel batch, espandi e propaga
        for i, node in enumerate(nodes):
            actions, priors, value = action_probs[i], priors_batch[i], values_batch[i].item()

            # Applica rumore di Dirichlet solo se stiamo espandendo la radice
            if node.parent is None:
                eps   = self.args.get("root_dir_eps", 0.3)
                alpha = self.args.get("root_dir_alpha", 0.3)
                noise = torch.distributions.Dirichlet(torch.full_like(priors, alpha)).sample()
                priors = priors * (1 - eps) + noise * eps

            # Aggiungi i figli al nodo
            for action, prior in zip(actions, priors):
                child_state = self.game.get_next_state(node.state, action)
                node.add_child(child_state, action=action, prior=prior.item())
            
            # Propaga il valore stimato dalla rete
            node.backpropagate(value)