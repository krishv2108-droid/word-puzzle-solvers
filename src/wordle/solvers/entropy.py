"""Information-theoretic solvers.

For a guess g and the remaining candidates C, the feedback pattern is a random variable
(uniform answer from C). Its distribution is the bucket sizes {n_p / |C|} over the 243
patterns. Two ways to rate g:
  * entropy  H = -sum p log2 p     - expected bits gained; the maximiser is `EntropySolver`.
  * worst-case  max_p n_p          - the biggest bucket you could be left with; the
                                     minimiser is `MinimaxSolver` (entropy breaks ties).
Guesses are drawn from the *whole* allowed list (or the hard-mode-legal subset), not just
candidates: a non-answer can split the candidates better than any answer. Among equally
good guesses we prefer one that is itself a candidate, since it also has a chance to win.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine import NUM_PATTERNS, WordleState
from .base import History, WordleData, WordleSolver

_EPS = 1e-9
_CHUNK = 2048  # guesses per block; bounds the int64 temporary in bincount (~38 MB)


@dataclass(frozen=True)
class Suggestion:
    word: str
    entropy: float  # expected information gain, bits
    expected_remaining: float  # E[#candidates left after this guess]
    worst_case: int  # largest bucket
    is_candidate: bool  # could this be the answer?


def bucket_counts(data: WordleData, pool: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """(len(pool), 243) counts: how many candidates yield each pattern for each pool guess."""
    out = np.empty((len(pool), NUM_PATTERNS), dtype=np.int32)
    cand_cols = data.matrix[:, cand]
    for s in range(0, len(pool), _CHUNK):
        rows = pool[s : s + _CHUNK]
        sub = cand_cols[rows].astype(np.int32)
        sub += (np.arange(len(rows), dtype=np.int32) * NUM_PATTERNS)[:, None]
        out[s : s + len(rows)] = np.bincount(
            sub.ravel(), minlength=len(rows) * NUM_PATTERNS
        ).reshape(len(rows), NUM_PATTERNS)
    return out


def entropy_bits(counts: np.ndarray) -> np.ndarray:
    n = counts.sum(1, keepdims=True)
    p = counts / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(counts > 0, p * np.log2(p), 0.0).sum(1)


class _InfoSolver(WordleSolver):
    """Common scoring; subclasses define what 'best' means."""

    def __init__(self, data: WordleData | None = None) -> None:
        super().__init__(data)
        self._first: str | None = None
        self.setup_seconds = 0.0
        self._warm()

    def _warm(self) -> None:
        """Cache the opening move: it is the same every game and the most expensive turn."""
        import time

        start = time.perf_counter()
        self._first = self._pick(self.data.legal_guess_mask((), False).nonzero()[0],
                                 np.arange(len(self.data.answers)))
        self.setup_seconds = time.perf_counter() - start

    def choose(self, state: WordleState, cand: np.ndarray) -> str:
        if state.guesses_made == 0 and self._first is not None:
            return self._first
        pool = self.data.legal_guess_mask(state.history, state.hard_mode).nonzero()[0]
        return self._pick(pool, cand)

    def _pick(self, pool: np.ndarray, cand: np.ndarray) -> str:
        counts = bucket_counts(self.data, pool, cand)
        key = self._key(counts)  # higher is better
        is_cand = np.isin(pool, self.data.answer_guess_idx[cand])
        best = np.flatnonzero(key >= key.max() - _EPS)
        preferred = best[is_cand[best]]
        pick = (preferred if len(preferred) else best)[0]  # sorted word list -> alphabetical
        return self.data.guesses[pool[pick]]

    def _key(self, counts: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def rank(
        self, history: History, hard_mode: bool = False, top_k: int = 10
    ) -> tuple[int, list[Suggestion]]:
        """Remaining candidate count and the top-k guesses (for the assistant UI)."""
        cand = self.data.candidates(history)
        if len(cand) == 0:
            return 0, []
        pool = self.data.legal_guess_mask(history, hard_mode).nonzero()[0]
        counts = bucket_counts(self.data, pool, cand)
        is_cand = np.isin(pool, self.data.answer_guess_idx[cand])
        key = self._key(counts)
        # best key first; candidates before non-candidates on ties; then alphabetical
        order = np.lexsort((np.arange(len(pool)), ~is_cand, -np.round(key / _EPS) * _EPS))[:top_k]
        h, n = entropy_bits(counts), len(cand)
        return n, [
            Suggestion(
                word=self.data.guesses[pool[i]],
                entropy=float(h[i]),
                expected_remaining=float((counts[i].astype(np.float64) ** 2).sum() / n),
                worst_case=int(counts[i].max()),
                is_candidate=bool(is_cand[i]),
            )
            for i in order
        ]


class EntropySolver(_InfoSolver):
    name = "entropy"

    def _key(self, counts: np.ndarray) -> np.ndarray:
        return entropy_bits(counts)


class MinimaxSolver(_InfoSolver):
    """Minimise the worst-case number of candidates left; break ties with entropy."""

    name = "minimax"

    def _key(self, counts: np.ndarray) -> np.ndarray:
        worst = counts.max(1).astype(np.float64)
        return -worst * 100.0 + entropy_bits(counts) / 10.0  # entropy only breaks ties
