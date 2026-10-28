"""Fuse the learned pair scorer with LLM-proposed candidate groups.

Group score = sum of the six learned pair logits (the log-odds the group is internally
"same-group") + bonus * LLM confidence, where the bonus applies only to groups the LLM
proposed. Exact search over all 2.6M partitions then picks the best global assignment.

`bonus` is the single tunable weight, in log-odds units per unit of LLM confidence: a group the
LLM proposed with confidence 1.0 gets `bonus` extra log-odds. bonus=0 is the learned scorer
alone. It is chosen on DEV puzzles using out-of-fold learned scores and never on TEST.
"""
from __future__ import annotations

import numpy as np

from ..common.llm import JSONClient
from ..common.solver import Solver
from .env import ConnectionsState, Proposal
from .learned import LearnedModel, logits_to_matrix
from .partition import best_partitions, group_id, pair_sum_scores
from .solvers.hybrid import CANDIDATES_SCHEMA, SYSTEM_PROMPT, HybridSolver
from .solvers.llm import user_prompt


def fused_partition(l120: np.ndarray, llm_scores: np.ndarray, bonus: float):
    """Best partition (word-index groups) for pair logits + LLM group scores."""
    group_scores = pair_sum_scores(logits_to_matrix(l120)) + bonus * llm_scores
    (_, parts), = best_partitions(group_scores, top_k=1)
    return parts, group_scores


def llm_group_scores(words, candidates: list[dict]) -> tuple[np.ndarray, int]:
    """(1820,) LLM confidence per group (0 if not proposed), #valid candidates."""
    # reuse the hybrid solver's candidate parsing; it does not touch the client for this
    return HybridSolver(client=None, model="", alpha=0.0).candidate_scores(words, candidates)


class FusedSolver(Solver[ConnectionsState, Proposal]):
    """Learned pair scorer + LLM candidate groups + exact search."""

    def __init__(self, client: JSONClient, model: str, learned: LearnedModel, bonus: float,
                 name: str = "learned + LLM") -> None:
        super().__init__()
        self.client, self.model, self.learned, self.bonus, self.name = client, model, learned, bonus, name
        self._m: dict[str, float] = {}

    def metrics(self) -> dict[str, float]:
        return self._m

    def next_action(self, state: ConnectionsState) -> Proposal:
        reply = self.client.complete_json(model=self.model, system=SYSTEM_PROMPT,
                                          user=user_prompt(state.words), schema=CANDIDATES_SCHEMA)
        for k in vars(reply.usage):
            setattr(self.usage, k, getattr(self.usage, k) + getattr(reply.usage, k))
        if reply.from_cache:
            self.usage.cached_latency_s += reply.latency_s
        llm_scores, valid = llm_group_scores(state.words, (reply.data or {}).get("candidates", []))
        parts, scores = fused_partition(self.learned.logits(state.words), llm_scores, self.bonus)
        gids = [group_id(g) for g in parts]
        self._m = {"llm_failure": float(reply.data is None), "candidates_valid": float(valid),
                   "groups_from_llm": float(sum(llm_scores[g] > 0 for g in gids)),
                   "llm_cache_hit": float(reply.from_cache)}
        return Proposal(
            groups=tuple(tuple(state.words[i] for i in g) for g in parts),
            confidences=tuple(float(1 / (1 + np.exp(-scores[g] / 6))) for g in gids),
        )
