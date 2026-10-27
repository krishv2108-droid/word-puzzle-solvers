"""Hybrid solver: the LLM proposes many candidate groups, constraint search picks the partition.

Why hybrid: asking an LLM for one partition commits it to a greedy reading of the red
herrings. Instead we ask for a *pool* of plausible groups, overlaps allowed, each with a
confidence, and let an exact search over all 2,627,625 partitions choose the single
assignment that uses every word exactly once and has the highest total score. A tempting
five-word cluster is therefore resolved by which four words leave the other 12 words
partitionable, not by which group the model happened to list first.

Group score = LLM confidence (if the group was proposed, else 0) + alpha * embedding
cohesion. alpha is fixed a priori (not tuned on the benchmark); alpha=0 gives the
"no embeddings" ablation, where unproposed groups score 0 and ties are broken arbitrarily.
"""
from __future__ import annotations

import numpy as np

from ...common.llm import JSONClient
from ...common.solver import Solver
from ..env import ConnectionsState, Proposal
from ..partition import all_groups, best_partitions, group_id, pair_sum_scores
from .baselines import EmbeddingModel
from .llm import BoardMatcher, user_prompt

SYSTEM_PROMPT = """You are an expert solver of the NYT Connections word puzzle.

You are given 16 words. The true solution partitions them into exactly 4 groups of 4 words \
that share a hidden category. Categories range from straightforward (members of a set) to \
tricky wordplay: words that precede or follow a common word (___ BOX), homophones, hidden \
smaller words, anagrams, and so on. Puzzles deliberately contain red herrings: words that \
fit more than one category.

Do NOT commit to a single partition. Propose up to 14 candidate groups of exactly 4 of the \
given words, including the tempting-but-possibly-wrong ones. A word may appear in several \
candidates. For each, give a short category label and a confidence between 0 and 1 that this \
exact set of 4 is one of the four true groups. Prefer fewer, sharper categories over vague \
ones; use low confidence for loose associations."""

CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "words": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
                "required": ["category", "words", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


class HybridSolver(Solver[ConnectionsState, Proposal]):
    def __init__(self, client: JSONClient, model: str, effort: str = "medium",
                 embedder: EmbeddingModel | None = None, alpha: float = 0.25) -> None:
        super().__init__()
        self.client, self.model, self.effort, self.alpha = client, model, effort, alpha
        self.embedder = embedder if alpha > 0 else None
        if alpha > 0 and self.embedder is None:
            self.embedder = EmbeddingModel("all-mpnet-base-v2")
        tag = "hybrid" if alpha > 0 else "hybrid, no embeddings"
        self.name = f"{tag} ({model.replace('claude-', '')})"
        self._m: dict[str, float] = {}

    def metrics(self) -> dict[str, float]:
        return self._m

    def candidate_scores(self, words, candidates: list[dict]) -> tuple[np.ndarray, int]:
        """(1820,) LLM confidence per group (max over duplicate proposals), #valid candidates."""
        matcher = BoardMatcher(words)
        index = {w: i for i, w in enumerate(words)}
        scores = np.zeros(len(all_groups()))
        valid = 0
        for c in candidates:
            mapped = [matcher.match(w) for w in c.get("words", [])]
            if len(mapped) != 4 or None in mapped or len(set(mapped)) != 4:
                continue
            valid += 1
            conf = min(1.0, max(0.0, float(c.get("confidence", 0) or 0)))
            gid = group_id([index[w] for w in mapped])
            scores[gid] = max(scores[gid], conf)
        return scores, valid

    def next_action(self, state: ConnectionsState) -> Proposal:
        reply = self.client.complete_json(
            model=self.model, system=SYSTEM_PROMPT, user=user_prompt(state.words),
            schema=CANDIDATES_SCHEMA, effort=self.effort,
        )
        for k in vars(reply.usage):
            setattr(self.usage, k, getattr(self.usage, k) + getattr(reply.usage, k))
        if reply.from_cache:
            self.usage.cached_latency_s += reply.latency_s
        candidates = (reply.data or {}).get("candidates", [])
        llm_scores, valid = self.candidate_scores(state.words, candidates)

        scores = llm_scores.copy()
        if self.embedder is not None:
            v = self.embedder.vectors(state.words)
            cohesion = np.clip(pair_sum_scores(v @ v.T) / 6, 0.0, 1.0)  # mean cosine in [0, 1]
            scores += self.alpha * cohesion
        (_, parts), = best_partitions(scores, top_k=1)

        labels = {}
        matcher = BoardMatcher(state.words)
        for c in candidates:
            mapped = [matcher.match(w) for w in c.get("words", [])]
            if len(mapped) == 4 and None not in mapped and len(set(mapped)) == 4:
                gid = group_id([state.words.index(w) for w in mapped])
                if llm_scores[gid] and gid not in labels:
                    labels[gid] = str(c.get("category", ""))
        gids = [group_id(g) for g in parts]
        self._m = {
            "llm_failure": float(reply.data is None),
            "candidates_valid": float(valid),
            "groups_from_llm": float(sum(llm_scores[g] > 0 for g in gids)),
            "llm_cache_hit": float(reply.from_cache),
        }
        return Proposal(
            groups=tuple(tuple(state.words[i] for i in g) for g in parts),
            labels=tuple(labels.get(g, "") for g in gids),
            confidences=tuple(float(llm_scores[g]) for g in gids),
        )
