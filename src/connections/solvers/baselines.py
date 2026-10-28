"""Non-LLM Connections baselines: a random floor and embedding similarity + exact search."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from ...common.cachefile import atomic_open
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

# Exact model snapshots the reported results were produced with. Pinning means a later upstream
# change to a model cannot silently change the embeddings (and invalidate the trained scorer).
MODEL_REVISIONS = {
    "all-mpnet-base-v2": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    "all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "BAAI/bge-large-en-v1.5": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
}


class EmbeddingModel:
    """Sentence-transformer wrapper that caches one unit vector per distinct word.

    Experiments persist the cache to disk (cache_dir) so repeated runs do not re-encode the
    vocabulary. A web server should pass cache_dir=None and a max_entries bound: it sees
    arbitrary user words, so an unbounded or disk-backed cache would grow forever and could fail
    on a read-only filesystem.
    """

    def __init__(self, model_name: str, cache_dir: Path | None = EMB_CACHE_DIR,
                 max_entries: int | None = None) -> None:
        self.model_name = model_name
        self.max_entries = max_entries
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        self._path = cache_dir / f"emb_{model_name.replace('/', '_')}.npz" if cache_dir else None
        if self._path and self._path.exists():
            try:
                z = np.load(self._path, allow_pickle=False)
                self._cache = dict(zip(z["words"].tolist(), z["vecs"], strict=True))
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                self._cache = {}  # truncated or corrupt cache: re-encode as needed

    def vectors(self, words) -> np.ndarray:
        missing = [w for w in dict.fromkeys(words) if w not in self._cache]
        if missing:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.model_name, device="cpu", revision=MODEL_REVISIONS.get(self.model_name))
            vecs = self._model.encode(
                [w.lower() for w in missing], normalize_embeddings=True, show_progress_bar=False
            )
            self._cache.update(zip(missing, vecs, strict=True))
            self._save()
        out = np.stack([self._cache[w] for w in words])  # build the result before any eviction
        if self.max_entries and len(self._cache) > self.max_entries:
            for key in list(self._cache)[: len(self._cache) - self.max_entries]:  # oldest first
                del self._cache[key]
        return out

    def _save(self) -> None:
        if not self._path:
            return
        try:
            with atomic_open(self._path) as f:
                np.savez(f, words=np.array(list(self._cache)), vecs=np.stack(list(self._cache.values())))
        except OSError:
            self._path = None  # read-only location: keep working in memory only


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
