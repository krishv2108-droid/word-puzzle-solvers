"""Non-LLM Connections baselines: a random floor and embedding similarity + exact search."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...common.solver import Solver
from ..env import ConnectionsState, Proposal
from ..partition import best_partitions, group_id, pair_sum_scores


class RandomSolver(Solver[ConnectionsState, Proposal]):
    """Floor: a uniformly random partition. Shows what 'no signal' scores."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def next_action(self, state: ConnectionsState) -> Proposal:
        order = self._rng.permutation(16)
        words = state.words
        return Proposal(tuple(tuple(words[i] for i in order[k : k + 4]) for k in range(0, 16, 4)))


EMB_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


class EmbeddingModel:
    """Sentence-transformer wrapper that caches one unit vector per distinct word (in memory,
    and on disk so repeated experiments do not re-encode the vocabulary)."""

    def __init__(self, model_name: str, cache_dir: Path | None = EMB_CACHE_DIR) -> None:
        self.model_name = model_name
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        self._path = cache_dir / f"emb_{model_name.replace('/', '_')}.npz" if cache_dir else None
        if self._path and self._path.exists():
            z = np.load(self._path, allow_pickle=False)
            self._cache = dict(zip(z["words"].tolist(), z["vecs"]))

    def vectors(self, words) -> np.ndarray:
        missing = [w for w in dict.fromkeys(words) if w not in self._cache]
        if missing:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device="cpu")
            vecs = self._model.encode(
                [w.lower() for w in missing], normalize_embeddings=True, show_progress_bar=False
            )
            self._cache.update(zip(missing, vecs))
            self._save()
        return np.stack([self._cache[w] for w in words])

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self._path, words=np.array(list(self._cache)), vecs=np.stack(list(self._cache.values())))


class EmbeddingSolver(Solver[ConnectionsState, Proposal]):
    """Embed each word, score every possible group by its summed pairwise cosine similarity,
    then exhaustively pick the partition with the highest total score.

    Known weakness (and the reason to compare against LLMs): single-word embeddings capture
    topical similarity, not wordplay ("___ box", hidden words, homophones) or red herrings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedder: EmbeddingModel | None = None):
        super().__init__()
        self.embedder = embedder or EmbeddingModel(model_name)
        self.name = f"embedding ({model_name})"

    def similarity(self, words) -> np.ndarray:
        v = self.embedder.vectors(words)
        return v @ v.T

    def next_action(self, state: ConnectionsState) -> Proposal:
        sim = self.similarity(state.words)
        group_scores = pair_sum_scores(sim)
        (_, parts), = best_partitions(group_scores, top_k=1)
        return Proposal(
            groups=tuple(tuple(state.words[i] for i in g) for g in parts),
            # mean pairwise cosine of each chosen group; a similarity, not a calibrated probability
            confidences=tuple(float(group_scores[group_id(g)] / 6) for g in parts),
        )
