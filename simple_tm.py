"""A minimal Python implementation of the Tsetlin Machine.

This module provides a simple multi-class Tsetlin Machine that can be stacked
into deeper architectures. The goal is readability rather than raw speed, so the
code favors clarity and numpy vectorisation where possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class TsetlinAutomaton:
    """Two-action Tsetlin Automaton.

    The automaton has ``num_states`` states where the first half selects action
    0 (literal excluded) and the second half selects action 1 (literal
    included). The ``state`` value is clipped to the valid range whenever it is
    updated.
    """

    num_states: int = 100
    state: int = field(init=False)

    def __post_init__(self) -> None:
        self.state = self.num_states // 2

    def action(self) -> int:
        """Return the current action: ``0`` for exclude and ``1`` for include."""
        return int(self.state >= self.num_states // 2)

    def update(self, reward: bool) -> None:
        """Update the automaton state based on the reward signal."""
        if reward:
            self.state = min(self.state + 1, self.num_states - 1)
        else:
            self.state = max(self.state - 1, 0)


@dataclass
class Clause:
    """Conjunctive clause consisting of a pair of automata per feature."""

    num_features: int
    num_states: int
    polarity: int = 1
    automata: List[TsetlinAutomaton] = field(init=False)

    def __post_init__(self) -> None:
        self.automata = [TsetlinAutomaton(self.num_states) for _ in range(self.num_features * 2)]

    def _literal_vector(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([x, 1 - x])

    def evaluate(self, x: np.ndarray) -> int:
        """Evaluate the clause on a binary feature vector ``x``."""
        literals = self._literal_vector(x)
        for ta, bit in zip(self.automata, literals):
            if ta.action() == 1 and bit == 0:
                return 0
        return 1

    def feedback(
        self,
        x: np.ndarray,
        target: bool,
        clause_output: int,
        feedback_type: str,
        s: float,
    ) -> None:
        """Apply feedback of the specified ``feedback_type`` to all automata."""
        literals = self._literal_vector(x)
        for ta, bit in zip(self.automata, literals):
            if feedback_type == "I":
                if target and clause_output == 1 and bit == 1:
                    ta.update(True)
                elif target and clause_output == 0 and bit == 1:
                    if np.random.random() < 1.0 / s:
                        ta.update(False)
            elif feedback_type == "II":
                if not target and clause_output == 1 and bit == 1:
                    ta.update(False)
            elif feedback_type == "III":
                if np.random.random() < 1.0 / (2 * s):
                    ta.update(True)


class MultiClassTsetlinMachine:
    """Basic multi-class Tsetlin Machine."""

    def __init__(
        self,
        num_classes: int,
        num_clauses_per_class: int,
        num_features: int,
        num_states: int = 100,
        T: int = 15,
        s: float = 3.9,
    ) -> None:
        self.num_classes = num_classes
        self.num_features = num_features
        self.T = T
        self.s = s
        self.clause_groups: List[List[Clause]] = []
        for _ in range(num_classes):
            clauses = [
                Clause(num_features, num_states, 1 if j % 2 == 0 else -1)
                for j in range(num_clauses_per_class)
            ]
            self.clause_groups.append(clauses)

    def _vote(self, x: np.ndarray) -> np.ndarray:
        votes = np.zeros(self.num_classes, dtype=int)
        for c, clauses in enumerate(self.clause_groups):
            for clause in clauses:
                votes[c] += clause.polarity * clause.evaluate(x)
        return votes

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> "MultiClassTsetlinMachine":
        for _ in range(epochs):
            for xi, yi in zip(X, y.astype(int)):
                votes = self._vote(xi)
                pred = np.argmax(votes)
                for class_index, clauses in enumerate(self.clause_groups):
                    target = class_index == yi
                    feedback_type = "I" if target else "II"
                    for clause in clauses:
                        out = clause.evaluate(xi)
                        clause.feedback(xi, target, out, feedback_type, self.s)
                for clauses in self.clause_groups:
                    for clause in clauses:
                        clause.feedback(xi, yi == pred, 0, "III", self.s)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([np.argmax(self._vote(x)) for x in X])


class DeepTsetlinMachine:
    """Stacked ensemble of multiple ``MultiClassTsetlinMachine`` layers."""

    def __init__(self, layer_configs: List[dict]) -> None:
        self.configs = layer_configs
        self.layers: List[MultiClassTsetlinMachine] = []

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> "DeepTsetlinMachine":
        data = X.copy()
        for cfg in self.configs:
            mctm = MultiClassTsetlinMachine(
                num_classes=cfg["num_classes"],
                num_clauses_per_class=cfg["num_clauses_per_class"],
                num_features=data.shape[1],
                num_states=cfg.get("num_states", 100),
                T=cfg.get("T", 15),
                s=cfg.get("s", 3.9),
            )
            mctm.fit(data, y, epochs)
            self.layers.append(mctm)
            votes = np.array([mctm._vote(row) for row in data])
            data = (votes >= 0).astype(int)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        data = X.copy()
        for layer in self.layers:
            votes = np.array([layer._vote(row) for row in data])
            data = (votes >= 0).astype(int)
        return self.layers[-1].predict(data)
