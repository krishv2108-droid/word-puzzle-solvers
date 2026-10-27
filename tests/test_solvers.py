import random

import numpy as np
import pytest

from src.common.evaluate import evaluate
from src.wordle.engine import (
    GRAY,
    GREEN,
    WordleEnv,
    WordleState,
    encode,
    hard_mode_violation,
    score_guess,
)
from src.wordle.patterns import build_pattern_matrix
from src.wordle.solvers.base import WordleData
from src.wordle.solvers.entropy import EntropySolver, MinimaxSolver, bucket_counts, entropy_bits
from src.wordle.solvers.frequency import FrequencySolver


@pytest.fixture(scope="module")
def data():
    return WordleData.load()


def test_pattern_matrix_matches_score_guess(data):
    rng = random.Random(0)
    for _ in range(20000):
        gi, ai = rng.randrange(len(data.guesses)), rng.randrange(len(data.answers))
        assert data.matrix[gi, ai] == encode(score_guess(data.guesses[gi], data.answers[ai]))


def test_pattern_matrix_duplicate_heavy_words():
    words = ["speed", "abide", "allee", "apple", "geese", "eerie", "otter", "tooth", "sassy", "aside"]
    P = build_pattern_matrix(words, words)
    for i, g in enumerate(words):
        for j, a in enumerate(words):
            assert P[i, j] == encode(score_guess(g, a)), (g, a)


def test_word_lists_have_expected_sizes(data):
    assert len(data.answers) == 2315 and len(data.guesses) == 12972
    assert set(data.answers) <= set(data.guesses)


def test_candidates_are_exactly_the_consistent_answers(data):
    rng = random.Random(1)
    for _ in range(30):
        secret = rng.choice(data.answers)
        history = []
        for _ in range(rng.randint(1, 3)):
            g = rng.choice(data.guesses)
            history.append((g, score_guess(g, secret)))
        cand = data.candidates(history)
        expected = [i for i, a in enumerate(data.answers)
                    if all(score_guess(g, a) == fb for g, fb in history)]
        assert list(cand) == expected
        assert data.answers.index(secret) in cand


def test_vectorised_hard_mode_mask_agrees_with_engine_rule(data):
    rng = random.Random(2)
    for _ in range(40):
        secret = rng.choice(data.answers)
        history = []
        for _ in range(rng.randint(1, 3)):
            g = rng.choice(data.guesses)
            history.append((g, score_guess(g, secret)))
        mask = data.legal_guess_mask(history, hard_mode=True)
        for i in rng.sample(range(len(data.guesses)), 300):
            assert mask[i] == (hard_mode_violation(data.guesses[i], history) is None), (
                data.guesses[i], history)
    assert data.legal_guess_mask([("crane", (GRAY,) * 5)], hard_mode=False).all()


def test_entropy_math():
    counts = np.array([[4, 0], [2, 2], [1, 3]])
    assert entropy_bits(counts) == pytest.approx([0.0, 1.0, 0.8112781])


def test_bucket_counts_partition_the_candidates(data):
    cand = np.arange(0, 2315, 7)
    pool = np.array([data.guess_index["crane"], data.guess_index["soare"]])
    counts = bucket_counts(data, pool, cand)
    assert counts.shape == (2, 243) and (counts.sum(1) == len(cand)).all()


def test_entropy_first_guess_matches_published_value():
    # Known result for the original lists: SOARE is the max-entropy opener at ~5.886 bits.
    solver = EntropySolver()
    n, top = solver.rank([], top_k=3)
    assert n == 2315
    assert top[0].word == "soare" == solver._first
    assert top[0].entropy == pytest.approx(5.886, abs=1e-3)
    assert [s.word for s in top][:2] == ["soare", "roate"]


def test_rank_reports_remaining_candidates_sorted_by_entropy():
    solver = EntropySolver()
    history = [("soare", score_guess("soare", "cigar"))]
    n, top = solver.rank(history, top_k=5)
    assert n == len(solver.data.candidates(history)) > 1
    assert all(top[i].entropy >= top[i + 1].entropy - 1e-9 for i in range(len(top) - 1))
    assert all(s.expected_remaining <= n and 1 <= s.worst_case <= n for s in top)


def test_rank_prefers_a_candidate_when_it_ties_a_non_candidate(data):
    # With exactly two candidates left, any guess that separates them scores 1 bit, but only
    # a candidate guess can also win outright, so the top suggestion must be a candidate.
    solver = EntropySolver(data)
    checked = 0
    for secret in data.answers[::40]:
        env = WordleEnv(data.guesses, max_guesses=10)
        state = env.reset(secret)
        while not env.done:
            if len(data.candidates(state.history)) == 2:
                n, top = solver.rank(state.history, top_k=1)
                assert n == 2 and top[0].is_candidate
                assert top[0].entropy == pytest.approx(1.0)
                checked += 1
                break
            state = env.step(solver.next_action(state))
    assert checked > 5


@pytest.mark.parametrize("cls", [FrequencySolver, EntropySolver, MinimaxSolver])
def test_solvers_solve_a_sample_without_illegal_guesses(cls, data):
    rng = random.Random(3)
    sample = rng.sample(data.answers, 60)
    env = WordleEnv(data.guesses, max_guesses=10)
    results = evaluate(cls(data), env, sample)
    assert all(r.solved for r in results)


@pytest.mark.parametrize("cls", [FrequencySolver, EntropySolver, MinimaxSolver])
def test_solvers_never_break_hard_mode(cls, data):
    # WordleEnv(hard_mode=True) raises on an illegal guess, so completing = never illegal.
    rng = random.Random(4)
    sample = rng.sample(data.answers, 60)
    env = WordleEnv(data.guesses, max_guesses=12, hard_mode=True)
    results = evaluate(cls(data), env, sample)
    assert len(results) == 60


def test_guesses_the_answer_once_one_candidate_remains(data):
    solver = EntropySolver(data)
    env = WordleEnv(data.guesses, max_guesses=10)
    state = env.reset("cigar")
    while not env.done:
        if len(data.candidates(state.history)) == 1:
            assert solver.next_action(state) == "cigar"
        state = env.step(solver.next_action(state))
    assert env.solved


def test_unsatisfiable_feedback_raises(data):
    solver = EntropySolver(data)
    bad = WordleState(history=(("crane", (GREEN,) * 4 + (GRAY,)), ("crane", (GRAY,) * 5)))
    with pytest.raises(ValueError, match="no answers"):
        solver.next_action(bad)


def test_minimax_worst_case_not_worse_than_entropy_choice(data):
    cand = np.arange(2315)
    pool = np.arange(len(data.guesses))
    counts = bucket_counts(data, pool, cand)
    e_first = data.guess_index[EntropySolver(data)._first]
    m_first = data.guess_index[MinimaxSolver(data)._first]
    assert counts[m_first].max() <= counts[e_first].max()
    assert counts[m_first].max() == counts.max(1).min()
