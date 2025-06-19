# node.py
import numpy as np
import math

class Node:
    """
    Nodo per MCTS.
      state          : copia di mask dopo l’azione che ha portato qui
      action_taken   : Tensor(2,)  [block_id, op]  (None per la radice)
      prior          : p_π(s,a)   (float)
    """
    def __init__(
        self,
        state,
        parent=None,
        action_taken=None,
        prior: float = 0.0,
    ):
        self.state         = state
        self.parent        = parent
        self.action_taken  = action_taken
        self.prior         = prior

        self.children: list[Node] = []

        self.visit_count   = 0
        self.value_sum     = 0.0

    # ------------------------------------------------------------------ #
    # UCB per la selezione
    # ------------------------------------------------------------------ #
    def ucb_score(self, c_puct: float):
        if self.visit_count == 0:
            q = 0.0
        else:
            # value in [-1,1] già normalizzato da tanh nel modello
            q = self.value_sum / self.visit_count
        u = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q + u

    # ------------------------------------------------------------------ #
    def select(self, c_puct: float):
        """Ritorna il figlio con UCB massimo."""
        return max(self.children, key=lambda n: n.ucb_score(c_puct))

    # ------------------------------------------------------------------ #
    def add_child(self, child_state, action, prior):
        child = Node(
            state=child_state,
            parent=self,
            action_taken=action,
            prior=prior,
        )
        self.children.append(child)
        return child

    # ------------------------------------------------------------------ #
    def backpropagate(self, value: float):
        """
        Propaga il valore (già dal punto di vista del giocatore radice)
        lungo la catena padre.  value ∈ [-1,1]
        """
        self.visit_count += 1
        self.value_sum   += value
        if self.parent is not None:
            self.parent.backpropagate(value)
