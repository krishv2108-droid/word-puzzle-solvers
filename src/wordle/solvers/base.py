"""Shared machinery for Wordle solvers: word tables, candidate filtering, hard-mode legality."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from ...common.solver import Solver
from ..engine import GRAY, GREEN, Feedback, WordleState, encode
from ..patterns import load_pattern_matrix, words_to_array
from ..words import load_allowed, load_answers

History = Sequence[tuple[str, Feedback]]


@dataclass
class WordleData:
    """Word lists plus the precomputed arrays every solver shares (built once)."""

    guesses: tuple[str, ...]  # every allowed guess (probe words), sorted
    answers: tuple[str, ...]  # possible secrets (targets), sorted
    matrix: np.ndarray  # (G, A) uint8 feedback codes
    guess_index: dict[str, int]
    answer_guess_idx: np.ndarray  # answer j -> its row in the guess list
    letters: np.ndarray  # (G, 5) letter indices
    letter_counts: np.ndarray  # (G, 26) letter multiplicities
    answer_letters: np.ndarray  # (A, 5)

    @classmethod
    @lru_cache(maxsize=None)
    def load(cls) -> "WordleData":
        return cls.build(load_allowed(), load_answers())

    @classmethod
    def build(cls, guesses: Sequence[str], answers: Sequence[str]) -> "WordleData":
        guesses, answers = tuple(guesses), tuple(answers)
        index = {w: i for i, w in enumerate(guesses)}
        letters = words_to_array(guesses)
        counts = np.zeros((len(guesses), 26), dtype=np.int8)
        for pos in range(letters.shape[1]):
            np.add.at(counts, (np.arange(len(guesses)), letters[:, pos]), 1)
        return cls(
            guesses=guesses,
            answers=answers,
            matrix=load_pattern_matrix(guesses, answers),
            guess_index=index,
            answer_guess_idx=np.array([index[a] for a in answers]),
            letters=letters,
            letter_counts=counts,
            answer_letters=words_to_array(answers),
        )

    def candidates(self, history: History) -> np.ndarray:
        """Indices (into `answers`) of every answer consistent with all feedback so far."""
        cand = np.arange(len(self.answers))
        for guess, fb in history:
            cand = cand[self.matrix[self.guess_index[guess], cand] == encode(fb)]
        return cand

    def legal_guess_mask(self, history: History, hard_mode: bool) -> np.ndarray:
        """Boolean mask over `guesses` of which words may be played next.

        Vectorised twin of engine.hard_mode_violation (tests check they agree).
        """
        mask = np.ones(len(self.guesses), dtype=bool)
        if not hard_mode:
            return mask
        for prev, fb in history:
            need = np.zeros(26, dtype=np.int8)
            for i, (ch, f) in enumerate(zip(prev, fb)):
                c = ord(ch) - 97
                if f == GREEN:
                    mask &= self.letters[:, i] == c
                if f != GRAY:
                    need[c] += 1
            for c in np.flatnonzero(need):
                mask &= self.letter_counts[:, c] >= need[c]
        return mask


class WordleSolver(Solver[WordleState, str]):
    """Base class: tracks the candidate set and delegates the choice to `choose`."""

    def __init__(self, data: WordleData | None = None) -> None:
        super().__init__()
        self.data = data or WordleData.load()

    def next_action(self, state: WordleState) -> str:
        cand = self.data.candidates(state.history)
        if len(cand) == 0:
            raise ValueError("no answers are consistent with the feedback so far")
        if len(cand) <= 2:  # nothing can beat guessing a candidate when it's 1 or 2 left
            return self.data.answers[cand[0]]
        return self.choose(state, cand)

    def choose(self, state: WordleState, cand: np.ndarray) -> str:
        raise NotImplementedError
