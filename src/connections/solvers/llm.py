"""LLM-based Connections solver: ask Claude for a full partition with category labels.

The model sees only the 16 words. Its JSON is mapped back onto the board words and, if it
is not a clean partition (a word reused, a word invented, a group of 3), repaired
deterministically rather than crashing; the number of repaired words is reported as a
metric so unreliable outputs are visible instead of silently patched.
"""
from __future__ import annotations

import re
from typing import Sequence

from ...common.llm import JSONClient
from ...common.solver import Solver
from ..env import ConnectionsState, Proposal

SYSTEM_PROMPT = """You are an expert solver of the NYT Connections word puzzle.

You are given 16 words. Partition them into exactly 4 groups of 4 words; the words in each \
group share a hidden category. Every word belongs to exactly one group.

Categories range from straightforward (members of a set) to tricky wordplay: words that \
precede or follow a common word (___ BOX), homophones, hidden smaller words, anagrams, \
words that change meaning with a prefix or suffix, and so on. Puzzles deliberately contain \
red herrings: a word that fits several categories, or five or more words that seem to \
belong together when only four do. Resolve these by finding the one assignment in which \
all 16 words are used exactly once.

For each group give a short category label and a confidence between 0 and 1 that the \
group is exactly right."""

PARTITION_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
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
    "required": ["groups"],
    "additionalProperties": False,
}


def user_prompt(words: Sequence[str]) -> str:
    return "Words:\n" + "\n".join(words)


def _norm(word: str) -> str:
    return re.sub(r"\s+", " ", word.strip().casefold())


def _loose(word: str) -> str:
    return re.sub(r"[^\w]", "", word.casefold()) or word.casefold()


class BoardMatcher:
    """Map free-text words from the model back to the exact board strings."""

    def __init__(self, board: Sequence[str]) -> None:
        self.exact = {_norm(w): w for w in board}
        self.loose = {_loose(w): w for w in board}

    def match(self, word: str) -> str | None:
        return self.exact.get(_norm(word)) or self.loose.get(_loose(word))


def build_partition(board: Sequence[str], groups: list[dict]) -> tuple[Proposal, int]:
    """Greedy-by-confidence strict partition from model groups; returns (proposal, #repaired).

    A model group is accepted only if it maps to exactly 4 distinct, still-unused board
    words. Whatever is left is chunked in board order ("repaired") so the result is always a
    valid partition and the failure is counted.
    """
    matcher = BoardMatcher(board)
    ranked = sorted(groups, key=lambda g: -float(g.get("confidence", 0) or 0))
    used: set[str] = set()
    accepted: list[tuple[tuple[str, ...], str, float]] = []
    for g in ranked:
        mapped = [matcher.match(w) for w in g.get("words", [])]
        if len(mapped) == 4 and None not in mapped and len(set(mapped)) == 4 and not (set(mapped) & used):
            accepted.append((tuple(mapped), str(g.get("category", "")), float(g.get("confidence", 0) or 0)))
            used |= set(mapped)
    leftover = [w for w in board if w not in used]
    for i in range(0, len(leftover), 4):
        accepted.append((tuple(leftover[i : i + 4]), "", 0.0))
    return (
        Proposal(
            groups=tuple(a[0] for a in accepted),
            labels=tuple(a[1] for a in accepted),
            confidences=tuple(min(1.0, max(0.0, a[2])) for a in accepted),
        ),
        len(leftover),
    )


class LLMSolver(Solver[ConnectionsState, Proposal]):
    def __init__(self, client: JSONClient, model: str, effort: str = "medium") -> None:
        super().__init__()
        self.client, self.model, self.effort = client, model, effort
        self.name = f"llm ({model.replace('claude-', '')})"
        self._m: dict[str, float] = {}

    def metrics(self) -> dict[str, float]:
        return self._m

    def next_action(self, state: ConnectionsState) -> Proposal:
        reply = self.client.complete_json(
            model=self.model, system=SYSTEM_PROMPT, user=user_prompt(state.words),
            schema=PARTITION_SCHEMA, effort=self.effort,
        )
        u = reply.usage
        for k in vars(u):
            setattr(self.usage, k, getattr(self.usage, k) + getattr(u, k))
        proposal, repaired = build_partition(state.words, (reply.data or {}).get("groups", []))
        self._m = {
            "repaired_words": float(repaired),
            "llm_failure": float(reply.data is None),
            "llm_cache_hit": float(reply.from_cache),
        }
        return proposal
