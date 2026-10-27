"""Wordle game engine: feedback scoring, hard-mode rules, and the evaluation environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..common.solver import Environment

GRAY, YELLOW, GREEN = 0, 1, 2
WORD_LEN = 5
NUM_PATTERNS = 3**WORD_LEN  # 243
ALL_GREEN = NUM_PATTERNS - 1  # 242

Feedback = tuple[int, ...]


def score_guess(guess: str, answer: str) -> Feedback:
    """Wordle feedback for `guess` against `answer`: a tuple of GRAY/YELLOW/GREEN.

    Duplicate letters are the classic bug. The rule is a two-pass allocation:
      1. Mark exact-position matches green and remove those letters from the pool of
         answer letters still available to be matched.
      2. Left to right, mark a non-green letter yellow only if an unconsumed copy of
         it remains in the pool (consuming it); otherwise gray.
    So a letter is never marked yellow/green more times than it occurs in the answer,
    and greens take priority over earlier yellows.
    """
    if len(guess) != WORD_LEN or len(answer) != WORD_LEN:
        raise ValueError(f"words must be {WORD_LEN} letters: {guess!r}, {answer!r}")
    result = [GRAY] * WORD_LEN
    pool: dict[str, int] = {}
    for g, a in zip(guess, answer):
        if g == a:
            continue
        pool[a] = pool.get(a, 0) + 1
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            result[i] = GREEN
    for i, g in enumerate(guess):
        if result[i] == GREEN:
            continue
        if pool.get(g, 0) > 0:
            result[i] = YELLOW
            pool[g] -= 1
    return tuple(result)


def encode(feedback: Sequence[int]) -> int:
    """Pack feedback into a base-3 integer 0..242 (first tile = most significant)."""
    code = 0
    for f in feedback:
        code = code * 3 + f
    return code


def decode(code: int) -> Feedback:
    digits = []
    for _ in range(WORD_LEN):
        code, d = divmod(code, 3)
        digits.append(d)
    return tuple(reversed(digits))


def feedback_to_str(feedback: Sequence[int]) -> str:
    return "".join(".yg"[f] for f in feedback)


def str_to_feedback(text: str) -> Feedback:
    """Inverse of feedback_to_str; '.' gray, 'y' yellow, 'g' green."""
    mapping = {".": GRAY, "y": YELLOW, "g": GREEN}
    if len(text) != WORD_LEN or any(c not in mapping for c in text):
        raise ValueError(f"feedback must be {WORD_LEN} chars of '.yg': {text!r}")
    return tuple(mapping[c] for c in text)


def is_consistent(candidate: str, guess: str, feedback: Sequence[int]) -> bool:
    """Could `candidate` be the answer given that `guess` produced `feedback`?"""
    return score_guess(guess, candidate) == tuple(feedback)


def hard_mode_violation(guess: str, history: Sequence[tuple[str, Feedback]]) -> str | None:
    """Return why `guess` breaks Wordle's hard-mode rule, or None if it is legal.

    Official rule: every revealed hint must be used. Greens stay in their position, and
    a letter revealed n times (green + yellow in one earlier guess) must appear at least
    n times. Note this is weaker than full consistency: hard mode does not forbid
    re-using gray letters or putting a yellow back in the same spot.
    """
    for prev, fb in history:
        needed: dict[str, int] = {}
        for i, (letter, f) in enumerate(zip(prev, fb)):
            if f == GREEN and guess[i] != letter:
                return f"position {i + 1} must be {letter.upper()}"
            if f != GRAY:
                needed[letter] = needed.get(letter, 0) + 1
        for letter, count in needed.items():
            if guess.count(letter) < count:
                return f"guess must contain {letter.upper()}" + (
                    f" ×{count}" if count > 1 else ""
                )
    return None


@dataclass(frozen=True)
class WordleState:
    """What the solver observes: its own guesses and the feedback received."""

    history: tuple[tuple[str, Feedback], ...] = ()
    hard_mode: bool = False

    @property
    def guesses_made(self) -> int:
        return len(self.history)


class WordleEnv(Environment[str, WordleState, str]):
    """One Wordle game. The puzzle is the answer word.

    `max_guesses` is a hard cap on turns. The benchmark uses a cap above the official 6
    so that average and worst-case guess counts are honest rather than truncated;
    win-rate-within-6 is derived from the step counts.
    """

    def __init__(
        self, allowed: Sequence[str], *, max_guesses: int = 6, hard_mode: bool = False
    ) -> None:
        self._allowed = frozenset(allowed)
        self.max_guesses = max_guesses
        self.hard_mode = hard_mode
        self._answer = ""
        self._state = WordleState(hard_mode=hard_mode)
        self._solved = False
        self.steps = 0

    @property
    def puzzle_id(self) -> str:
        return self._answer

    @property
    def solved(self) -> bool:
        return self._solved

    @property
    def done(self) -> bool:
        return self._solved or self.steps >= self.max_guesses

    def reset(self, puzzle: str) -> WordleState:
        self._answer = puzzle
        self._solved = False
        self.steps = 0
        self._state = WordleState(hard_mode=self.hard_mode)
        return self._state

    def step(self, action: str) -> WordleState:
        if self.done:
            raise RuntimeError("game is over")
        if action not in self._allowed:
            raise ValueError(f"{action!r} is not in the allowed guess list")
        if self.hard_mode:
            violation = hard_mode_violation(action, self._state.history)
            if violation:
                raise ValueError(f"hard mode violation for {action!r}: {violation}")
        fb = score_guess(action, self._answer)
        self.steps += 1
        self._solved = action == self._answer
        self._state = WordleState(self._state.history + ((action, fb),), self.hard_mode)
        return self._state
