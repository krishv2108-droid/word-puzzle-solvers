"""Memorization probe: does the model already know these puzzles?

For each puzzle we give the model only its public puzzle number and date and ask it to
recall the four category names. It never sees the 16 words. A model that has not seen the
puzzle can only guess, and will rarely reproduce the exact (often quirky) category names, so
matching several true labels is evidence of memorization.

Chance baseline (control): the same recalled labels are scored against OTHER puzzles' true
labels. Generic labels ("___ BALL", "COLORS") will match by luck at some rate; the control
measures that rate so the flag threshold is read against it, not against zero.

A puzzle is flagged "likely seen" when >= FLAG_AT of its 4 labels are matched. This is a
conservative, one-sided test: flagged = probably in training data; unflagged does NOT prove
a puzzle is unseen (the model can know a puzzle without being able to recite it from a date).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..common.llm import JSONClient
from .data import Puzzle

FLAG_AT = 2
MATCH_RATIO = 0.85
CONTROL_SAMPLES = 5

SYSTEM_PROMPT = (
    "You are asked to recall a past New York Times Connections puzzle. Each puzzle has four "
    "hidden categories. Give your best recollection of the four category names for the puzzle "
    "identified by its number and date. Always answer with exactly four category names, even "
    "if you are unsure."
)
SCHEMA = {
    "type": "object",
    "properties": {"categories": {"type": "array", "items": {"type": "string"}}},
    "required": ["categories"],
    "additionalProperties": False,
}


def _clean(label: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", label.casefold())).strip()


def _same_label(a: str, b: str) -> bool:
    """Close in spelling AND sharing at least half their words (plural-insensitive), so
    "board games" does not count as recalling "card games"."""
    if SequenceMatcher(None, a, b).ratio() < MATCH_RATIO:
        return False
    ta, tb = ({t.rstrip("s") for t in x.split()} for x in (a, b))
    return len(ta & tb) / max(1, len(ta | tb)) >= 0.5 or len(ta) == len(tb) == 1


def labels_matched(recalled: list[str], truth: list[str]) -> int:
    """Number of true labels matched one-to-one by some recalled label."""
    remaining = [_clean(r) for r in recalled]
    hits = 0
    for t in map(_clean, truth):
        best = max(
            (r for r in remaining if _same_label(r, t)),
            key=lambda r: SequenceMatcher(None, r, t).ratio(),
            default=None,
        )
        if best is not None:
            hits += 1
            remaining.remove(best)
    return hits


@dataclass
class ProbeResult:
    puzzle_id: str
    matched: int  # true labels recalled (0-4)
    control_matched: float  # mean matches against other puzzles' labels (chance level)
    flagged: bool


def run_probe(client: JSONClient, model: str, puzzles: list[Puzzle], seed: int = 0,
              effort: str = "low") -> list[ProbeResult]:
    rng = random.Random(seed)
    recalled: dict[int, list[str]] = {}
    for p in puzzles:
        reply = client.complete_json(
            model=model, system=SYSTEM_PROMPT,
            user=f"NYT Connections puzzle #{p.id}, published {p.date}.",
            schema=SCHEMA, effort=effort, max_tokens=4000,
        )
        recalled[p.id] = [str(c) for c in (reply.data or {}).get("categories", [])]
    results = []
    for p in puzzles:
        truth = [g.label for g in p.groups]
        others = rng.sample([q for q in puzzles if q.id != p.id], min(CONTROL_SAMPLES, len(puzzles) - 1))
        control = sum(labels_matched(recalled[p.id], [g.label for g in q.groups]) for q in others)
        m = labels_matched(recalled[p.id], truth)
        results.append(ProbeResult(str(p.id), m, control / max(1, len(others)), m >= FLAG_AT))
    return results


def probe_summary(results: list[ProbeResult]) -> dict[str, float]:
    n = len(results)
    return {
        "n": n,
        "mean_labels_matched": sum(r.matched for r in results) / n,
        "mean_labels_matched_control": sum(r.control_matched for r in results) / n,
        "flagged_rate": sum(r.flagged for r in results) / n,
    }
