"""Baseline: letter-frequency / positional-frequency heuristic.

Among the *remaining candidates* (so every guess could be the answer), play the word whose
letters are most common among the other candidates:

    score(w) = sum_i  count(candidates with w[i] at position i)        # positional
             + sum_{distinct letters c in w}  count(candidates containing c)   # presence

Repeated letters count once in the presence term (a second 'e' adds little information).
No notion of how the feedback will *split* the candidates - that's what entropy adds.
"""
from __future__ import annotations

import numpy as np

from ..engine import WordleState
from .base import WordleSolver


class FrequencySolver(WordleSolver):
    name = "frequency"

    def choose(self, state: WordleState, cand: np.ndarray) -> str:
        L = self.data.answer_letters[cand]  # (n, 5)
        positional = np.zeros((5, 26))
        for i in range(5):
            positional[i] = np.bincount(L[:, i], minlength=26)
        presence = np.zeros(26)
        for c in range(26):
            presence[c] = (L == c).any(1).sum()
        first_copy = np.ones_like(L, dtype=bool)
        for i in range(1, 5):
            first_copy[:, i] = ~(L[:, :i] == L[:, i : i + 1]).any(1)
        score = positional[np.arange(5), L].sum(1) + (presence[L] * first_copy).sum(1)
        return self.data.answers[cand[int(np.argmax(score))]]  # argmax: ties -> alphabetical
