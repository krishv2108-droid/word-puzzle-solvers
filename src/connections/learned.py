"""Learned pair scorer: predict "do these two words share a group?" and search for the best split.

Model: gradient-boosted trees on pair features (embedding similarities in the context of the
other 14 words, plus string features for wordplay). Prediction: p(pair in same group).

Why summing pair log-odds is the right objective: if each of the 120 word pairs were
independent with probability p_ij of being "same group", the likelihood of a partition is
    prod_{same} p_ij * prod_{different} (1 - p_ij)
        = prod_all (1 - p_ij) * prod_{same} p_ij / (1 - p_ij).
The first product does not depend on the partition, so the most likely partition maximises
the sum of logit(p_ij) over same-group pairs: exactly what `pair_sum_scores` computes when
fed a matrix of pair logits. The independence assumption is wrong (pairs are strongly
coupled), but the exact search enforces the one global constraint that matters: a valid
partition into four groups of four.

Features are computed per puzzle, so they are *relative*: the same cosine means different
things in a puzzle of food words and a puzzle of verbs. Everything is symmetrised (min/max).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..common.solver import Solver
from .env import ConnectionsState, Proposal
from .partition import all_groups, best_partitions, group_id, pair_sum_scores
from .solvers.baselines import EmbeddingModel

PAIRS = np.array(list(combinations(range(16), 2)))  # (120, 2)
I, J = PAIRS[:, 0], PAIRS[:, 1]


def _ranks(sim: np.ndarray) -> np.ndarray:
    """rank[i, j] = position of j among i's neighbours by similarity (0 = nearest)."""
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    order = np.argsort(-s, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(16)[:, None], order] = np.arange(16)[None, :]
    return rank.astype(float)


def embedding_features(vecs: np.ndarray, tag: str) -> dict[str, np.ndarray]:
    """Per-pair features from one embedding model, raw and centred on the puzzle's mean."""
    out: dict[str, np.ndarray] = {}
    centred = vecs - vecs.mean(axis=0)
    centred /= np.linalg.norm(centred, axis=1, keepdims=True) + 1e-9
    for name, v in (("", vecs), ("c_", centred)):
        sim = v @ v.T
        off = sim[~np.eye(16, dtype=bool)]
        rank = _ranks(sim)
        masked = sim.copy()
        np.fill_diagonal(masked, 0)
        prof = np.corrcoef(masked)  # similarity of the two words' similarity profiles
        out[f"{tag}_{name}cos"] = sim[I, J]
        out[f"{tag}_{name}z"] = (sim[I, J] - off.mean()) / (off.std() + 1e-9)
        out[f"{tag}_{name}rank_min"] = np.minimum(rank[I, J], rank[J, I])
        out[f"{tag}_{name}rank_max"] = np.maximum(rank[I, J], rank[J, I])
        out[f"{tag}_{name}prof"] = np.nan_to_num(prof[I, J])
    return out


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _bigrams(w: str) -> set[str]:
    return {w[k : k + 2] for k in range(len(w) - 1)} or {w}


def string_features(words: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Wordplay-oriented pair features computed from spelling alone."""
    lw = [w.lower() for w in words]
    cols: dict[str, list[float]] = {k: [] for k in (
        "len_min", "len_max", "len_diff", "same_first", "same_last", "prefix", "suffix", "substring",
        "letter_jaccard", "bigram_jaccard", "anagram", "both_palindrome", "any_palindrome",
        "both_space", "both_hyphen", "any_hyphen", "both_digit", "any_digit", "both_symbol",
        "any_symbol", "shared_letters")}
    for i, j in PAIRS:
        a, b = lw[i], lw[j]
        la, lb = sorted((len(a), len(b)))
        cols["len_min"].append(la), cols["len_max"].append(lb), cols["len_diff"].append(lb - la)
        cols["same_first"].append(float(a[:1] == b[:1]))
        cols["same_last"].append(float(a[-1:] == b[-1:]))
        cols["prefix"].append(_common_prefix(a, b))
        cols["suffix"].append(_common_prefix(a[::-1], b[::-1]))
        cols["substring"].append(float(a in b or b in a))
        sa, sb = set(a), set(b)
        cols["letter_jaccard"].append(len(sa & sb) / max(1, len(sa | sb)))
        ba, bb = _bigrams(a), _bigrams(b)
        cols["bigram_jaccard"].append(len(ba & bb) / max(1, len(ba | bb)))
        cols["anagram"].append(float(sorted(a) == sorted(b)))
        pa, pb = a == a[::-1] and len(a) > 2, b == b[::-1] and len(b) > 2
        cols["both_palindrome"].append(float(pa and pb)), cols["any_palindrome"].append(float(pa or pb))
        cols["both_space"].append(float(" " in a and " " in b))
        cols["both_hyphen"].append(float("-" in a and "-" in b)), cols["any_hyphen"].append(float("-" in a or "-" in b))
        da, db = any(c.isdigit() for c in a), any(c.isdigit() for c in b)
        cols["both_digit"].append(float(da and db)), cols["any_digit"].append(float(da or db))
        ya, yb = not any(c.isalnum() for c in a), not any(c.isalnum() for c in b)
        cols["both_symbol"].append(float(ya and yb)), cols["any_symbol"].append(float(ya or yb))
        cols["shared_letters"].append(sum((Counter(a) & Counter(b)).values()))
    return {f"str_{k}": np.array(v, dtype=float) for k, v in cols.items()}


def puzzle_features(words: tuple[str, ...], embedders: dict[str, EmbeddingModel],
                    use_string: bool = True) -> dict[str, np.ndarray]:
    feats: dict[str, np.ndarray] = {}
    for tag, emb in embedders.items():
        feats.update(embedding_features(emb.vectors(words), tag))
    if use_string:
        feats.update(string_features(words))
    return feats


def same_group_labels(puzzle) -> np.ndarray:
    level = {w: g.level for g in puzzle.groups for w in g.words}
    return np.array([level[puzzle.words[i]] == level[puzzle.words[j]] for i, j in PAIRS], dtype=float)


def feature_matrix(puzzles, embedders, use_string: bool = True):
    """(n_puzzles, 120, F) features, (n_puzzles, 120) labels, feature names."""
    rows, labels, names = [], [], None
    for p in puzzles:
        f = puzzle_features(p.words, embedders, use_string)
        names = names or list(f)
        rows.append(np.stack([f[k] for k in names], axis=1))
        labels.append(same_group_labels(p))
    return np.stack(rows), np.stack(labels), names


def logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_pair_model(X: np.ndarray, y: np.ndarray, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
        min_samples_leaf=40, random_state=seed)
    return model.fit(X.reshape(-1, X.shape[-1]), y.reshape(-1))


def pair_logits(model, X: np.ndarray) -> np.ndarray:
    """(n_puzzles, 120) logit(p same group) for a fitted model."""
    return logit(model.predict_proba(X.reshape(-1, X.shape[-1]))[:, 1]).reshape(X.shape[:2])


def logits_to_matrix(l120: np.ndarray) -> np.ndarray:
    m = np.zeros((16, 16))
    m[I, J] = l120
    m[J, I] = l120
    return m


def solve_from_pair_scores(l120: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    """Best partition (word-index groups) and the (1820,) group scores it was chosen from."""
    group_scores = pair_sum_scores(logits_to_matrix(l120))
    (_, parts), = best_partitions(group_scores, top_k=1)
    return parts, group_scores


def groups_correct(puzzle, parts: list[list[int]]) -> int:
    truth = {g.words for g in puzzle.groups}
    return sum(frozenset(puzzle.words[i] for i in g) in truth for g in parts)


def cross_validate(puzzles, X, y, columns: list[int], folds: int = 5, seed: int = 0) -> np.ndarray:
    """Per-puzzle groups_correct from K-fold CV by puzzle (train on other folds' pairs)."""
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(len(puzzles)) % folds
    correct = np.zeros(len(puzzles))
    for k in range(folds):
        tr, te = fold_of != k, fold_of == k
        model = fit_pair_model(X[tr][:, :, columns], y[tr], seed)
        logits = pair_logits(model, X[te][:, :, columns])
        for idx, l in zip(np.flatnonzero(te), logits):
            parts, _ = solve_from_pair_scores(l)
            correct[idx] = groups_correct(puzzles[idx], parts)
    return correct


@dataclass
class LearnedModel:
    model: object
    embedders: dict[str, EmbeddingModel]
    names: list[str]
    columns: list[int]
    use_string: bool = True

    def logits(self, words: tuple[str, ...]) -> np.ndarray:
        f = puzzle_features(words, self.embedders, self.use_string)
        X = np.stack([f[k] for k in self.names], axis=1)[None][:, :, self.columns]
        return pair_logits(self.model, X)[0]


class LearnedSolver(Solver[ConnectionsState, Proposal]):
    """Learned pair scorer + exact partition search. No LLM, so ~50 ms per puzzle."""

    def __init__(self, learned: LearnedModel, name: str = "learned pair scorer") -> None:
        super().__init__()
        self.learned, self.name = learned, name

    def next_action(self, state: ConnectionsState) -> Proposal:
        l120 = self.learned.logits(state.words)
        parts, scores = solve_from_pair_scores(l120)
        conf = [1 / (1 + np.exp(-scores[group_id(g)] / 6)) for g in parts]  # sigmoid(mean pair logit)
        return Proposal(
            groups=tuple(tuple(state.words[i] for i in g) for g in parts),
            confidences=tuple(float(c) for c in conf),
        )
