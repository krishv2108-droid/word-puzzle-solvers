"""Exact search over every way to split 16 words into 4 groups of 4.

There are C(16,4) = 1,820 possible groups and 16!/(4!^4 * 4!) = 2,627,625 partitions. That
is small enough to enumerate once, so "search" becomes one vectorised gather-and-sum: score
all 2.6M partitions at once from a per-group score array and take the best. This is exact
(no heuristic pruning) and handles constraints for free: give a forbidden group a score of
-inf and no partition containing it can win. The hybrid solver uses exactly that to pick a
globally consistent partition from LLM-proposed groups.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np

from ..common.cachefile import load_npy, save_npy

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "partitions16.npy"
N_WORDS = 16
N_PARTITIONS = 2_627_625  # 16! / (4!^4 * 4!)


@lru_cache(maxsize=None)
def all_groups() -> np.ndarray:
    """(1820, 4) word indices of every possible group, in lexicographic order."""
    return np.array(list(combinations(range(N_WORDS), 4)), dtype=np.int8)


@lru_cache(maxsize=None)
def group_ids() -> dict[tuple[int, ...], int]:
    return {tuple(g): i for i, g in enumerate(all_groups().tolist())}


@lru_cache(maxsize=None)
def all_partitions() -> np.ndarray:
    """(2627625, 4) group ids; each row is one partition into four groups."""
    cached = load_npy(CACHE, shape=(N_PARTITIONS, 4))
    if cached is not None:
        return cached
    ids = group_ids()
    out = []
    for g1 in combinations(range(1, N_WORDS), 3):
        a = (0,) + g1
        rest1 = [w for w in range(N_WORDS) if w not in a]
        for g2 in combinations(rest1[1:], 3):  # group containing the lowest remaining word
            b = (rest1[0],) + g2
            rest2 = [w for w in rest1 if w not in b]
            for g3 in combinations(rest2[1:], 3):
                c = (rest2[0],) + g3
                d = tuple(w for w in rest2 if w not in c)
                out.append((ids[a], ids[b], ids[c], ids[d]))
    parts = np.array(out, dtype=np.uint16)
    save_npy(CACHE, parts)
    return parts


def group_id(indices) -> int:
    return group_ids()[tuple(sorted(int(i) for i in indices))]


def pair_sum_scores(sim: np.ndarray) -> np.ndarray:
    """Score each of the 1,820 groups as the sum of its 6 pairwise similarities."""
    g = all_groups().astype(np.intp)
    total = np.zeros(len(g))
    for i, j in combinations(range(4), 2):
        total += sim[g[:, i], g[:, j]]
    return total


def best_partitions(group_scores: np.ndarray, top_k: int = 1) -> list[tuple[float, list[list[int]]]]:
    """Top-k partitions by total group score: [(score, [[word idx x4] x4]), ...], best first."""
    parts = all_partitions()
    totals = group_scores[parts].sum(axis=1)
    k = min(top_k, len(totals))
    top = np.argpartition(-totals, k - 1)[:k]
    top = top[np.argsort(-totals[top], kind="stable")]
    groups = all_groups()
    return [(float(totals[i]), [groups[g].tolist() for g in parts[i]]) for i in top]
