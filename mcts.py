import torch
from node import Node
from graphviz import Digraph


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
        
        # La chiamata a get_encoded_state usa la storia del gioco principale, che è una semplificazione
        # ma è consistente con il design attuale del tuo codice.
        encoded_states = torch.stack([self.game.get_encoded_state(n.state) for n in nodes])
        scalars = torch.stack([self.game.get_scalar() for n in nodes]) # Scalari sono uguali per tutti i nodi di una ricerca
        
        # Esegui una sola inferenza per l'intero batch
        action_probs, priors_batch, values_batch = self.model.fwd_infer(encoded_states, scalars, top_k=self.args.get("top_k", 32))

        # Per ogni nodo nel batch, espandi e propaga
        for i, node in enumerate(nodes):
            actions, priors, value = action_probs[i], priors_batch[i], values_batch[i].item()

            # Applica rumore di Dirichlet solo se stiamo espandendo la radice
            #if node.parent is None:
            #    eps   = self.args.get("root_dir_eps", 0.3)
            #    alpha = self.args.get("root_dir_alpha", 0.3)
            #    
            #    if self.args["device"] == "mps":
            #        # 1. Costruiamo il tensore di concentrazioni su CPU
            #        concentration_cpu = torch.full_like(priors, alpha, device="cpu")
            #        # 2. Campioniamo su CPU
            #        noise_cpu = torch.distributions.Dirichlet(concentration_cpu).sample()
            #        # 3. Spostiamo il rumore sul device di priors
            #        noise = noise_cpu.to(priors.device)
            #    else:
            #        noise = torch.distributions.Dirichlet(torch.full_like(priors, alpha)).sample()
            #    priors = priors * (1 - eps) + noise * eps

            # Aggiungi i figli al nodo
            for action, prior in zip(actions, priors):
                child_state = self.game.get_next_state(node.state, action)
                node.add_child(child_state, action=action, prior=prior.item())
            
            # Propaga il valore stimato dalla rete
            node.backpropagate(value)

    def render_mcts_tree(
        self,
        root: Node,
        filename: str = "mcts_tree",
        max_depth: int = 3,
        draw_q: bool = True,
        palette: tuple[str, ...] = (
            "#dfe7fd", "#c7d7fd", "#bdd2ff", "#a6c1ff", "#8fb0ff",
            "#779fff", "#5e8eff", "#447cff", "#2a6aff", "#0f58ff"
        ),
        orientation: str = "TB",
        engine: str = "dot",
        dpi: int = 300,
        ranksep: float = 1.0,
        nodesep: float = 0.5,
    ):
        """
        Disegna e salva un sotto-albero MCTS completo, mostrando tutti i nodi fino a `max_depth`.

        Args
        ----
        root        : nodo radice (ad es. mcts.last_root)
        filename    : basename (senza estensione) del file di output
        max_depth   : profondità massima da disegnare (ROOT=0)
        draw_q      : se True mostra Q-value medio in ogni nodo
        palette     : colori alternati per livelli di profondità
        orientation : 'TB' (verticale) o 'LR' (orizzontale)
        engine      : Graphviz layout engine (es. 'dot','twopi','circo')
        dpi         : risoluzione immagine in DPI
        ranksep     : spazio verticale fra livelli
        nodesep     : spazio orizzontale fra nodi
        """
        dot = Digraph(comment="MCTS-Tree", engine=engine)
        dot.attr(rankdir=orientation, splines="polyline")
        dot.graph_attr.update(
            dpi=str(dpi),
            ranksep=str(ranksep),
            nodesep=str(nodesep),
        )

        def add_node(node: Node, depth: int = 0):
            # Mostra tutti i nodi finché depth <= max_depth
            if depth > max_depth:
                return
            nid = str(id(node))
            # Costruisci label con info complete
            if node.action_taken is None:
                label = "ROOT"
            else:
                b, o = node.action_taken
                # visit_count, prior e Q-value
                N = node.visit_count
                P = node.prior
                Q = (node.value_sum / N) if N else 0.0
                label = f"b={b} o={o} | N={N} | P={P:.2f}"
                if draw_q:
                    label += f" | Q={Q:.2f}"
            dot.node(
                nid,
                label=label,
                shape="box",
                style="filled,rounded",
                fontsize="10",
                fontname="Helvetica",
                fillcolor=palette[depth % len(palette)],
            )
            for child in node.children:
                cid = str(id(child))
                dot.edge(nid, cid)
                add_node(child, depth + 1)

        add_node(root, 0)

        fmt = filename.split('.')[-1] if '.' in filename else 'png'
        dot.render(filename, format=fmt, cleanup=True)
        print(f"✅ Albero completo salvato in {filename}.{fmt}")
