"""Game-agnostic interfaces shared by the Wordle and Connections modules.

A game is modelled as an `Environment` that hides the ground truth and exposes an
observable *state*; a `Solver` maps state -> action. The evaluation runner only
depends on these two interfaces, so any new game or approach plugs into the same
benchmark and report code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

P = TypeVar("P")  # puzzle (ground truth the solver must not see)
S = TypeVar("S")  # observable state handed to the solver
A = TypeVar("A")  # action the solver submits


@dataclass
class Usage:
    """Cumulative LLM usage. Non-LLM solvers leave it at zero."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    api_calls: int = 0

    def snapshot(self) -> "Usage":
        return Usage(**vars(self))

    def minus(self, earlier: "Usage") -> "Usage":
        return Usage(**{k: v - getattr(earlier, k) for k, v in vars(self).items()})


class Environment(ABC, Generic[P, S, A]):
    """One episode of a game. Hides the puzzle's answer from the solver."""

    steps: int = 0

    @property
    @abstractmethod
    def puzzle_id(self) -> str: ...

    @property
    @abstractmethod
    def done(self) -> bool: ...

    @property
    @abstractmethod
    def solved(self) -> bool: ...

    @abstractmethod
    def reset(self, puzzle: P) -> S:
        """Start a new episode and return the initial observable state."""

    @abstractmethod
    def step(self, action: A) -> S:
        """Apply an action; increments `steps`; returns the new state."""

    def metrics(self) -> dict[str, float]:
        """Optional per-episode numeric metrics (averaged by the report)."""
        return {}


class Solver(ABC, Generic[S, A]):
    name: str = "solver"

    def __init__(self) -> None:
        self.usage = Usage()

    def reset(self) -> None:
        """Called before each episode. Expensive one-off setup belongs in __init__."""

    @abstractmethod
    def next_action(self, state: S) -> A: ...
