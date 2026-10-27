"""Connections as an evaluation environment.

Protocol: the solver sees only the 16 words and submits ONE complete partition (four groups
of four, optionally with labels and confidences). We score how many groups match the truth
exactly, and per difficulty tier. This one-shot protocol is deliberate: it makes results
comparable across approaches and cheap to run. The real game allows four mistakes with
feedback ("one away"); that interactive variant is a possible extension.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..common.solver import Environment
from .data import TIERS, Puzzle


@dataclass(frozen=True)
class ConnectionsState:
    words: tuple[str, ...]


@dataclass(frozen=True)
class Proposal:
    groups: tuple[tuple[str, ...], ...]
    labels: tuple[str, ...] | None = None  # category guesses, parallel to `groups`
    confidences: tuple[float, ...] | None = None  # in [0, 1], parallel to `groups`


def validate_partition(groups, words) -> None:
    flat = [w for g in groups for w in g]
    if len(groups) != 4 or any(len(g) != 4 for g in groups) or sorted(flat) != sorted(words):
        raise ValueError("proposal must partition the 16 board words into 4 groups of 4")


class ConnectionsEnv(Environment[Puzzle, ConnectionsState, Proposal]):
    def __init__(self) -> None:
        self._puzzle: Puzzle | None = None
        self._correct = 0
        self._tier_hit = [False] * 4
        self._done = False
        self.steps = 0

    @property
    def puzzle_id(self) -> str:
        return str(self._puzzle.id)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def solved(self) -> bool:
        return self._correct == 4

    def reset(self, puzzle: Puzzle) -> ConnectionsState:
        self._puzzle = puzzle
        self._correct, self._tier_hit, self._done, self.steps = 0, [False] * 4, False, 0
        return ConnectionsState(puzzle.words)

    def step(self, action: Proposal) -> ConnectionsState:
        if self._done:
            raise RuntimeError("episode is over")
        validate_partition(action.groups, self._puzzle.words)
        proposed = {frozenset(g) for g in action.groups}
        for group in self._puzzle.groups:
            self._tier_hit[group.level] = group.words in proposed
        self._correct = sum(self._tier_hit)
        self.steps, self._done = 1, True
        return ConnectionsState(self._puzzle.words)

    def metrics(self) -> dict[str, float]:
        out = {"groups_correct": float(self._correct)}
        out.update({f"tier_{TIERS[i]}": float(hit) for i, hit in enumerate(self._tier_hit)})
        return out
