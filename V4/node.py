# node.py
from __future__ import annotations
import math, random
from typing import Optional, Dict, List

class Node:
    def __init__(self, state, parent: Optional["Node"] = None, action_taken: Optional[int] = None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: List[Node] = []
        self.child_map: Dict[int, Node] = {}
        self.depth = 0 if parent is None else parent.depth + 1

    def ucb_score(self, c_puct: float) -> float:
        q = 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count
        parent_visits = self.parent.visit_count if self.parent is not None else 1
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1.0 + self.visit_count)
        return q + u + (1e-6 * random.random())  # jitter per tie-break

    def select(self, c_puct: float) -> "Node":
        return max(self.children, key=lambda n: n.ucb_score(c_puct))

    def add_child(self, child_state, action: int, prior: float) -> "Node":
        child = Node(state=child_state, parent=self, action_taken=action, prior=prior)
        self.children.append(child)
        self.child_map[action] = child
        return child

    def backpropagate(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backpropagate(value)

    def q_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self) -> str:
        return (
            f"Node(depth={self.depth}, action={self.action_taken}, "
            f"prior={self.prior:.3f}, N={self.visit_count}, Q={self.q_value():.3f})"
        )
