import pytest

from src.wordle.engine import (
    WordleEnv,
    hard_mode_violation,
    score_guess,
    str_to_feedback,
)

ALLOWED = ["crane", "slate", "speed", "abide", "apple", "allee", "irate", "trace", "crate", "grace"]


def history(*pairs):
    return [(g, str_to_feedback(f)) for g, f in pairs]


# --- hard mode --------------------------------------------------------------------

def test_hard_mode_no_history_is_always_legal():
    assert hard_mode_violation("crane", []) is None


def test_hard_mode_greens_must_stay_in_place():
    h = history(("crane", "g...."))  # C is green at position 1
    assert hard_mode_violation("crate", h) is None
    assert "position 1" in hard_mode_violation("slate", h)


def test_hard_mode_yellows_must_be_included_anywhere():
    h = history(("crane", ".y..."))  # R is somewhere
    assert hard_mode_violation("irate", h) is None
    assert "R" in hard_mode_violation("slate", h)


def test_hard_mode_allows_reusing_gray_letters_and_yellow_in_same_spot():
    # Official hard mode only forces *using* hints; it does not ban known-bad letters.
    h = history(("crane", ".y..."))
    assert hard_mode_violation("crane", h) is None


def test_hard_mode_counts_duplicate_hints():
    # Two E's revealed (one green, one yellow) -> next guess needs two E's.
    h = history(("speed", "..gy."))
    assert hard_mode_violation("abide", h) is not None  # only one E, wrong spot too
    assert hard_mode_violation("eerie", h) is not None  # E count ok but pos 3 not E
    assert hard_mode_violation("geese", h) is None


def test_hard_mode_checks_every_previous_guess():
    h = history(("crane", "g...."), ("slate", "...y."))
    assert hard_mode_violation("crate", h) is None  # C fixed at 1, T included
    assert hard_mode_violation("grace", h) is not None  # no T


# --- environment ------------------------------------------------------------------

def test_env_win_flow_and_hidden_answer():
    env = WordleEnv(ALLOWED, max_guesses=6)
    state = env.reset("crate")
    assert state.history == () and not env.done
    state = env.step("slate")
    assert not env.solved and state.guesses_made == 1
    assert state.history[0] == ("slate", score_guess("slate", "crate"))
    env.step("crate")
    assert env.solved and env.done and env.steps == 2


def test_env_loses_at_cap():
    env = WordleEnv(ALLOWED, max_guesses=2)
    env.reset("crate")
    env.step("slate")
    env.step("irate")
    assert env.done and not env.solved
    with pytest.raises(RuntimeError):
        env.step("crate")


def test_env_rejects_unknown_word():
    env = WordleEnv(ALLOWED)
    env.reset("crate")
    with pytest.raises(ValueError):
        env.step("zzzzz")


def test_env_enforces_hard_mode():
    env = WordleEnv(ALLOWED, hard_mode=True)
    env.reset("crate")
    env.step("crane")  # C R A green, N E: E green? crane vs crate -> ggg.g
    with pytest.raises(ValueError, match="hard mode"):
        env.step("slate")  # drops the green C R A


def test_env_reset_clears_state():
    env = WordleEnv(ALLOWED)
    env.reset("crate")
    env.step("slate")
    state = env.reset("apple")
    assert state.history == () and env.steps == 0 and not env.solved
