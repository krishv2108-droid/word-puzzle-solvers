"""Precomputed feedback-pattern matrix.

P[g, a] = base-3 feedback code (0..242) for guessing word g when the answer is a.
Once built (12,972 x 2,315 uint8, ~30 MB), scoring a guess against every remaining
candidate is a numpy row slice, and the feedback distribution is one bincount, so
entropy over all guesses costs milliseconds instead of minutes of Python loops.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from ..common.cachefile import load_npy, save_npy
from .engine import WORD_LEN

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
_POW3 = 3 ** np.arange(WORD_LEN - 1, -1, -1)  # first tile is most significant, as in engine.encode


def words_to_array(words: Sequence[str]) -> np.ndarray:
    """(N, 5) uint8 array of letter indices a=0..z=25."""
    return np.frombuffer("".join(words).encode(), dtype=np.uint8).reshape(len(words), WORD_LEN) - 97


def build_pattern_matrix(
    guesses: Sequence[str], answers: Sequence[str], chunk: int = 256
) -> np.ndarray:
    """Vectorised version of engine.score_guess for every (guess, answer) pair.

    Same two-pass rule, expressed per guess position i: a non-green letter is yellow iff
    the number of *earlier* non-green copies of it in the guess is smaller than the number
    of non-green copies of it in the answer (those are the answer letters not consumed by
    greens). Tests check this against score_guess.
    """
    G, A = words_to_array(guesses), words_to_array(answers)
    out = np.empty((len(G), len(A)), dtype=np.uint8)
    for start in range(0, len(G), chunk):
        g = G[start : start + chunk]  # (c, 5)
        green = g[:, None, :] == A[None, :, :]  # (c, A, 5)
        code = np.zeros(green.shape[:2], dtype=np.int32)
        for i in range(WORD_LEN):
            same_letter = A[None, :, :] == g[:, None, i, None]  # (c, A, 5)
            available = (same_letter & ~green).sum(-1)  # answer copies not used by greens
            if i:
                earlier = (
                    (g[:, None, :i] == g[:, None, i, None]) & ~green[:, :, :i]
                ).sum(-1)  # earlier non-green guess copies of this letter
            else:
                earlier = 0
            yellow = ~green[:, :, i] & (earlier < available)
            code += (green[:, :, i] * 2 + yellow) * _POW3[i]
        out[start : start + chunk] = code
    return out


def load_pattern_matrix(
    guesses: Sequence[str], answers: Sequence[str], cache_dir: Path = CACHE_DIR
) -> np.ndarray:
    """Build the matrix, or load it from an on-disk cache keyed by the word lists."""
    key = hashlib.sha1(("\n".join(guesses) + "|" + "\n".join(answers)).encode()).hexdigest()[:16]
    path = cache_dir / f"patterns_{len(guesses)}x{len(answers)}_{key}.npy"
    cached = load_npy(path, shape=(len(guesses), len(answers)))
    if cached is not None:
        return cached
    matrix = build_pattern_matrix(guesses, answers)
    save_npy(path, matrix)  # best effort: a read-only filesystem just rebuilds next time
    return matrix
