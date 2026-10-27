"""Calibrated confidence for each proposed group.

A raw model score is not a probability. For each group in the chosen partition we compute two
signals and map them to P(this group is exactly right) with a logistic regression fitted on
out-of-fold DEV predictions:

  * mean pair logit  - how strongly the scorer believes the four words belong together
  * gap              - how much worse the best partition becomes if this group is forbidden
                       (a max-marginal: a large gap means the group is forced by the other words,
                       a small gap means a red herring could swap in)

Reliability is checked on TEST (see experiments.py --calibration).
"""
from __future__ import annotations

import numpy as np

from .learned import groups_correct, logits_to_matrix, solve_from_pair_scores
from .partition import all_groups, best_partitions, group_id, pair_sum_scores


def constrain(scores: np.ndarray, solved=(), wrong=()) -> np.ndarray:
    """Group scores with already-found groups locked in and known-wrong groups forbidden.

    Any other group touching a solved word is forbidden, so every valid partition must contain
    the solved groups themselves.
    """
    out = scores.copy()
    if solved:
        used = [w for g in solved for w in g]
        out[np.isin(all_groups(), used).any(axis=1)] = -np.inf
        for g in solved:
            out[group_id(g)] = scores[group_id(g)]
    for g in wrong:
        out[group_id(g)] = -np.inf
    return out


def group_signals(l120: np.ndarray, solved=(), wrong=()) -> tuple[list[list[int]], np.ndarray]:
    """Best partition plus an (4, 2) array of [mean pair logit, gap] for its groups.

    Solved groups get gap=inf (they are certain). Raises ValueError if the constraints leave
    no valid partition.
    """
    scores = constrain(pair_sum_scores(logits_to_matrix(l120)), solved, wrong)
    (best, parts), = best_partitions(scores, top_k=1)
    if not np.isfinite(best):
        raise ValueError("these constraints leave no valid partition")
    solved_ids = {group_id(g) for g in solved}
    feats = []
    for g in parts:
        gid = group_id(g)
        if gid in solved_ids:
            feats.append([scores[gid] / 6.0, np.inf])
            continue
        banned = scores.copy()
        banned[gid] = -np.inf
        (alt, _), = best_partitions(banned, top_k=1)
        feats.append([scores[gid] / 6.0, best - alt if np.isfinite(alt) else 50.0])
    return parts, np.array(feats)


def collect(puzzles, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_groups, 2) signals and (n_groups,) 0/1 'group is exactly right' labels."""
    X, y = [], []
    for p, l in zip(puzzles, logits):
        parts, feats = group_signals(l)
        truth = {g.words for g in p.groups}
        for g, f in zip(parts, feats):
            X.append(f)
            y.append(float(frozenset(p.words[i] for i in g) in truth))
    return np.array(X), np.array(y)


class Calibrator:
    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(C=1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Calibrator":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, edges=(0, 0.1, 0.2, 0.35, 0.5, 1.0001)):
    """[(lo, hi, n, mean predicted, observed rate)] and expected calibration error."""
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append((lo, min(hi, 1.0), int(m.sum()), float(p[m].mean()), float(y[m].mean())))
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return rows, float(ece)
