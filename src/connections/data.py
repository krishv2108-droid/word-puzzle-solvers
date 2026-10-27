"""NYT Connections puzzles: data model and loader for the Hugging Face dev dataset.

Run `python data/download_connections.py` first. The data is NYT content and is gitignored.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "connections" / "Connections_Data.csv"
TIERS = ("yellow", "green", "blue", "purple")  # Group Level 0..3, easiest to hardest


@dataclass(frozen=True)
class Group:
    label: str
    level: int  # 0 = yellow ... 3 = purple
    words: frozenset[str]

    @property
    def tier(self) -> str:
        return TIERS[self.level]


@dataclass(frozen=True)
class Puzzle:
    id: int
    date: str  # YYYY-MM-DD
    words: tuple[str, ...]  # the 16 words in the order they were laid out on the board
    groups: tuple[Group, ...]  # ground truth, sorted by level

    def group_of(self, word: str) -> Group:
        return next(g for g in self.groups if word in g.words)


def load_puzzles(path: Path = DATA_PATH) -> list[Puzzle]:
    """Load and validate every puzzle, oldest first. Raises on any malformed puzzle."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python data/download_connections.py`")
    by_game: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_game[int(row["Game ID"])].append(row)
    puzzles = [_build_puzzle(gid, rows) for gid, rows in by_game.items()]
    return sorted(puzzles, key=lambda p: (p.date, p.id))


def _build_puzzle(game_id: int, rows: list[dict]) -> Puzzle:
    if len(rows) != 16:
        raise ValueError(f"puzzle {game_id}: expected 16 words, got {len(rows)}")
    board = sorted(rows, key=lambda r: (int(r["Starting Row"]), int(r["Starting Column"])))
    words = tuple(r["Word"] for r in board)
    if len(set(words)) != 16:
        raise ValueError(f"puzzle {game_id}: duplicate words")
    grouped: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[int(r["Group Level"])].append(r)
    if sorted(grouped) != [0, 1, 2, 3] or any(len(v) != 4 for v in grouped.values()):
        raise ValueError(f"puzzle {game_id}: need four groups of four across levels 0-3")
    groups = tuple(
        Group(label=grouped[lv][0]["Group Name"], level=lv,
              words=frozenset(r["Word"] for r in grouped[lv]))
        for lv in range(4)
    )
    return Puzzle(game_id, rows[0]["Puzzle Date"], words, groups)
