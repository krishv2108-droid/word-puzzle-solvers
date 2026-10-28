"""Wordle logic behind the API: input validation, suggestions, and full-game simulation.

Kept free of FastAPI types so it can be unit-tested and reused. Solvers are stateless w.r.t.
a request: the whole game is the history of (guess, feedback) pairs, so the server holds no
session state and any number of concurrent users are independent.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict

import numpy as np

from ..wordle.engine import WordleEnv, feedback_to_str
from ..wordle.solvers.base import WordleData
from ..wordle.solvers.entropy import EntropySolver, MinimaxSolver, bucket_counts, entropy_bits
from ..wordle.solvers.frequency import FrequencySolver

GRAY_ALIASES = ".bx_-"
MAX_CANDIDATES_SHOWN = 30


class InputError(ValueError):
    """Bad user input (maps to HTTP 422)."""


def parse_feedback(text: str) -> tuple[int, ...]:
    t = text.strip().lower()
    if len(t) != 5 or any(c not in GRAY_ALIASES + "yg" for c in t):
        raise InputError(f"feedback must be 5 characters from '.', 'y', 'g' (got {text!r})")
    return tuple(2 if c == "g" else 1 if c == "y" else 0 for c in t)


class WordleService:
    def __init__(self, data: WordleData | None = None) -> None:
        self.data = data or WordleData.load()
        self.rankers = {"entropy": EntropySolver(self.data), "minimax": MinimaxSolver(self.data)}
        self.players = {**self.rankers, "frequency": FrequencySolver(self.data)}
        self._rank_cache: dict[tuple, tuple] = {}

    # ---- validation ----------------------------------------------------------------------

    def parse_history(self, entries: list[tuple[str, str]]) -> list[tuple[str, tuple[int, ...]]]:
        history = []
        for guess, fb in entries:
            g = guess.strip().lower()
            if len(g) != 5 or not g.isalpha():
                raise InputError(f"guess must be 5 letters (got {guess!r})")
            if g not in self.data.guess_index:
                raise InputError(f"{g!r} is not in Wordle's allowed guess list")
            history.append((g, parse_feedback(fb)))
        return history

    # ---- assistant mode ------------------------------------------------------------------

    def suggest(self, entries: list[tuple[str, str]], hard_mode: bool = False,
                solver: str = "entropy", top_k: int = 10) -> dict:
        if solver not in self.rankers:
            raise InputError(f"solver must be one of {sorted(self.rankers)}")
        history = self.parse_history(entries)
        cand = self.data.candidates(history)
        remaining = len(cand)
        words = [self.data.answers[i] for i in cand]
        base = {
            "remaining": remaining,
            "bits_remaining": math.log2(remaining) if remaining else 0.0,
            "candidates": words[:MAX_CANDIDATES_SHOWN],
            "candidates_truncated": remaining > MAX_CANDIDATES_SHOWN,
            "total_answers": len(self.data.answers),
            "hard_mode": hard_mode,
            "solver": solver,
        }
        if remaining == 0:
            return {**base, "suggestions": [], "message":
                    "No possible answer matches this feedback. Check the guesses and colours for a typo. "
                    "If they are right, the answer may be a newer word: this tool knows the 2,315 original Wordle answers."}
        if remaining == 1:
            return {**base, "suggestions": [self._single(history, words[0], hard_mode)], "message":
                    f"Only one word fits: {words[0].upper()}."}
        key = (solver, hard_mode, top_k, tuple((g, f) for g, f in history))
        if key not in self._rank_cache:
            if len(self._rank_cache) > 512:
                self._rank_cache.clear()
            self._rank_cache[key] = self.rankers[solver].rank(history, hard_mode, top_k)
        _, ranked = self._rank_cache[key]
        return {**base, "suggestions": [asdict(s) for s in ranked], "message": None}

    def _single(self, history, word: str, hard_mode: bool) -> dict:
        return {"word": word, "entropy": 0.0, "expected_remaining": 1.0, "worst_case": 1,
                "is_candidate": True}

    def _guess_entropy(self, history, guess: str) -> float:
        cand = self.data.candidates(history)
        counts = bucket_counts(self.data, np.array([self.data.guess_index[guess]]), cand)
        return float(entropy_bits(counts)[0])

    # ---- simulation ----------------------------------------------------------------------

    def simulate(self, answer: str | None = None, solver: str = "entropy", hard_mode: bool = False,
                 max_guesses: int = 10, rng: random.Random | None = None) -> dict:
        if solver not in self.players:
            raise InputError(f"solver must be one of {sorted(self.players)}")
        if answer is None:
            answer = (rng or random).choice(self.data.answers)
        answer = answer.strip().lower()
        if answer not in set(self.data.answers):
            raise InputError(f"{answer!r} is not one of the {len(self.data.answers)} possible answers")
        player = self.players[solver]
        env = WordleEnv(self.data.guesses, max_guesses=max_guesses, hard_mode=hard_mode)
        state = env.reset(answer)
        steps = []
        while not env.done:
            guess = player.next_action(state)
            before = len(self.data.candidates(state.history))
            expected_bits = self._guess_entropy(state.history, guess) if before > 1 else 0.0
            state = env.step(guess)
            after = len(self.data.candidates(state.history))
            steps.append({
                "guess": guess,
                "feedback": feedback_to_str(state.history[-1][1]),
                "remaining_before": before,
                "remaining_after": after,
                "expected_bits": expected_bits,
                "actual_bits": math.log2(before / after) if after else 0.0,
            })
        return {"answer": answer, "solved": env.solved, "guesses": env.steps, "solver": solver,
                "hard_mode": hard_mode, "steps": steps}
