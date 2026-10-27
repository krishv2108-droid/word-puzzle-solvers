"""Duplicate-letter feedback is the classic Wordle bug, so this is tested three ways:
hand-verified cases, an independent reference implementation cross-checked on
exhaustive/random inputs, and algebraic properties."""
import itertools
import random
from collections import Counter

import pytest

from src.wordle.engine import (
    ALL_GREEN,
    GRAY,
    GREEN,
    NUM_PATTERNS,
    YELLOW,
    decode,
    encode,
    feedback_to_str,
    is_consistent,
    score_guess,
    str_to_feedback,
)


def fb(guess: str, answer: str) -> str:
    return feedback_to_str(score_guess(guess, answer))


# --- hand-verified cases (g = green, y = yellow, . = gray) -------------------------

@pytest.mark.parametrize(
    "guess, answer, expected",
    [
        ("crane", "crane", "ggggg"),
        # answer has one E (last); only the first guessed E earns a yellow
        ("speed", "abide", "..y.y"),
        # 'apple' has one L: first guessed L yellow, second gray; A and E exact
        ("allee", "apple", "gy..g"),
        # two E's exact; the remaining answer E (position 1 of 'eerie') feeds the third
        ("geese", "eerie", ".gy.g"),
        ("eerie", "geese", "yg..g"),
        # two T's and two O's, none in place
        ("otter", "tooth", "yyy.."),
        ("tooth", "otter", "yy.y."),
        # extra copies beyond what the answer holds are gray; greens are the exact ones
        ("aaaaa", "banal", ".g.g."),
        # one S and one A in the answer: first S yellow, later S's gray
        ("sassy", "aside", "yy..."),
        # three E's in the answer, all in place; surplus guessed E's are gray
        ("eeeee", "geese", ".gg.g"),
        ("eeeee", "elbow", "g...."),
    ],
)
def test_known_duplicate_letter_cases(guess, answer, expected):
    assert fb(guess, answer) == expected


def test_green_consumes_the_letter_before_an_earlier_misplaced_copy():
    # 'abbey' has ONE E (position 4). In 'ebbed' the E at position 1 is misplaced and the
    # E at position 4 is exact. The exact one is green, so the early E must NOT be yellow.
    assert fb("ebbed", "abbey") == ".ggg."


# --- independent reference implementation -----------------------------------------

def reference_score(guess: str, answer: str):
    """Different algorithm: per-letter bookkeeping instead of one global pool."""
    out = [GRAY] * 5
    for letter in set(guess):
        gpos = [i for i in range(5) if guess[i] == letter]
        apos = [i for i in range(5) if answer[i] == letter]
        greens = [i for i in gpos if guess[i] == answer[i]]
        for i in greens:
            out[i] = GREEN
        spare_in_answer = len(apos) - len(greens)
        non_green = [i for i in gpos if i not in greens]
        for i in non_green[: max(0, spare_in_answer)]:  # leftmost get the yellows
            out[i] = YELLOW
    return tuple(out)


def test_matches_reference_exhaustive_small_alphabet():
    # 2-letter and 3-letter alphabets force heavy letter repetition
    words = ["".join(p) for p in itertools.product("ab", repeat=5)]
    for g in words:
        for a in words:
            assert score_guess(g, a) == reference_score(g, a), (g, a)
    words3 = ["".join(p) for p in itertools.product("abc", repeat=5)]
    rng = random.Random(0)
    for _ in range(30000):
        g, a = rng.choice(words3), rng.choice(words3)
        assert score_guess(g, a) == reference_score(g, a), (g, a)


def test_matches_reference_on_real_words():
    from src.wordle.words import load_allowed

    allowed = load_allowed()
    rng = random.Random(1)
    for _ in range(20000):
        g, a = rng.choice(allowed), rng.choice(allowed)
        assert score_guess(g, a) == reference_score(g, a), (g, a)


# --- properties -------------------------------------------------------------------

def test_identity_is_all_green():
    for w in ["crane", "eerie", "mamma", "fuzzy"]:
        assert score_guess(w, w) == (GREEN,) * 5


def test_marks_never_exceed_answer_letter_count():
    rng = random.Random(2)
    words = ["".join(p) for p in itertools.product("abc", repeat=5)]
    for _ in range(5000):
        g, a = rng.choice(words), rng.choice(words)
        marked = Counter(letter for letter, f in zip(g, score_guess(g, a)) if f != GRAY)
        have = Counter(a)
        assert all(marked[c] <= have[c] for c in marked)
        # and a letter is fully "explained" when both words contain it
        for c in marked:
            assert marked[c] == min(g.count(c), a.count(c))


def test_green_iff_same_position():
    rng = random.Random(3)
    words = ["".join(p) for p in itertools.product("abc", repeat=5)]
    for _ in range(2000):
        g, a = rng.choice(words), rng.choice(words)
        f = score_guess(g, a)
        assert all((f[i] == GREEN) == (g[i] == a[i]) for i in range(5))


def test_encode_decode_roundtrip_covers_all_243_patterns():
    codes = set()
    for p in itertools.product((GRAY, YELLOW, GREEN), repeat=5):
        c = encode(p)
        assert decode(c) == p
        codes.add(c)
    assert codes == set(range(NUM_PATTERNS))
    assert encode((GREEN,) * 5) == ALL_GREEN
    assert encode((GRAY,) * 5) == 0


def test_feedback_string_roundtrip_and_validation():
    assert str_to_feedback("gy.yg") == (GREEN, YELLOW, GRAY, YELLOW, GREEN)
    assert feedback_to_str(str_to_feedback("gy.yg")) == "gy.yg"
    with pytest.raises(ValueError):
        str_to_feedback("gyxyg")
    with pytest.raises(ValueError):
        str_to_feedback("gyg")


def test_rejects_wrong_length():
    with pytest.raises(ValueError):
        score_guess("cranes", "crane")


def test_is_consistent():
    assert is_consistent("abide", "speed", str_to_feedback("..y.y"))
    assert not is_consistent("speed", "speed", str_to_feedback("..y.y"))
