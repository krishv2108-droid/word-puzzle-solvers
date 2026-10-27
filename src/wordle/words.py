"""Word lists.

Two lists, on purpose:
  * answers - the 2,315 words Wordle can pick as the secret. These are the *targets*,
    and what a solver may treat as the candidate set.
  * allowed - answers plus 10,657 obscure words that are valid guesses but never
    answers (12,972 total). A non-answer can still be the best *probe* because it can
    split the candidates more evenly than any answer, so solvers may guess from here.
Run `python data/download_wordle.py` to fetch them.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "wordle"


def _read(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python data/download_wordle.py`")
    return tuple(path.read_text().split())


@lru_cache(maxsize=None)
def load_answers(path: Path | None = None) -> tuple[str, ...]:
    return _read(path or DATA_DIR / "answers.txt")


@lru_cache(maxsize=None)
def load_allowed(path: Path | None = None) -> tuple[str, ...]:
    return _read(path or DATA_DIR / "allowed.txt")
