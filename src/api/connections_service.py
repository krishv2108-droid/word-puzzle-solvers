"""Connections logic behind the API: validation, solving with calibrated confidence, and hints."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ..connections.calibration import Calibrator, group_signals
from ..connections.learned import LearnedModel
from ..connections.partition import group_id
from ..connections.solvers.baselines import EmbeddingModel
from .wordle_service import InputError

MAX_WORD_LEN = 40


class ConnectionsService:
    def __init__(self, learned: LearnedModel, calibrator: Calibrator) -> None:
        self.learned, self.calibrator = learned, calibrator
        self._lock = threading.Lock()  # the embedding models are not thread-safe

    @classmethod
    def load(cls, path: Path) -> "ConnectionsService":
        import joblib

        b = joblib.load(path)
        embs = {tag: EmbeddingModel(name) for tag, name in b["embedders"].items()}
        return cls(LearnedModel(b["model"], embs, b["names"], b["cols"]), b["calibrator"])

    # ---- validation ---------------------------------------------------------------------

    @staticmethod
    def _key(w: str) -> str:
        return " ".join(w.split()).casefold()

    def clean_words(self, words: list[str]) -> tuple[str, ...]:
        cleaned = tuple(" ".join(w.split()).upper() for w in words)
        if len(cleaned) != 16:
            raise InputError(f"enter exactly 16 words (got {len(cleaned)})")
        if any(not w for w in cleaned):
            raise InputError("every one of the 16 boxes needs a word")
        if any(len(w) > MAX_WORD_LEN for w in cleaned):
            raise InputError(f"words must be at most {MAX_WORD_LEN} characters")
        if len({self._key(w) for w in cleaned}) != 16:
            raise InputError("the 16 words must all be different")
        return cleaned

    def _group_indices(self, board: tuple[str, ...], groups: list[list[str]], what: str) -> list[list[int]]:
        index = {self._key(w): i for i, w in enumerate(board)}
        out = []
        for g in groups:
            try:
                idx = sorted(index[self._key(w)] for w in g)
            except KeyError as e:
                raise InputError(f"a {what} group contains a word that is not on the board: {e.args[0]!r}")
            if len(idx) != 4 or len(set(idx)) != 4:
                raise InputError(f"each {what} group must be 4 different board words")
            out.append(idx)
        return out

    # ---- analysis -----------------------------------------------------------------------

    def _analyse(self, board, solved_idx, wrong_idx):
        with self._lock:
            l120 = self.learned.logits(board)
        try:
            parts, feats = group_signals(l120, solved_idx, wrong_idx)
        except ValueError as e:
            raise InputError(str(e) + ". Check the groups you marked as solved or wrong.")
        solved_ids = {group_id(g) for g in solved_idx}
        conf = []
        for g, f in zip(parts, feats):
            conf.append(1.0 if group_id(g) in solved_ids else float(self.calibrator.predict(f[None, :])[0]))
        groups = [{"words": [board[i] for i in g], "confidence": c, "solved": group_id(g) in solved_ids}
                  for g, c in zip(parts, conf)]
        return sorted(groups, key=lambda x: -x["confidence"])

    def solve(self, words: list[str]) -> dict:
        board = self.clean_words(words)
        groups = self._analyse(board, [], [])
        return {"groups": groups, "expected_correct": sum(g["confidence"] for g in groups),
                "note": "Confidence is the estimated probability that the group is exactly right."}

    def hint(self, words: list[str], solved: list[list[str]] | None = None,
             wrong: list[list[str]] | None = None) -> dict:
        board = self.clean_words(words)
        solved_idx = self._group_indices(board, solved or [], "solved")
        wrong_idx = self._group_indices(board, wrong or [], "wrong")
        if len(solved_idx) > 4 or len({w for g in solved_idx for w in g}) != 4 * len(solved_idx):
            raise InputError("solved groups must be distinct and must not overlap")
        groups = self._analyse(board, solved_idx, wrong_idx)
        open_groups = [g for g in groups if not g["solved"]]
        if not open_groups:
            return {"group": None, "remaining_groups": 0, "message": "All four groups are already solved."}
        best = open_groups[0]
        return {"group": best, "remaining_groups": len(open_groups), "message": None}
