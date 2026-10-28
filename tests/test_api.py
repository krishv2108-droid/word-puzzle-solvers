import random

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.wordle.engine import feedback_to_str, hard_mode_violation, score_guess


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def suggest(client, history, **kw):
    return client.post("/api/wordle/suggest", json={"history": history, **kw})


def test_health_and_info(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    info = client.get("/api/wordle/info").json()
    assert info["answers"] == 2315 and info["allowed_guesses"] == 12972
    assert {"entropy", "minimax", "frequency"} <= set(info["solvers"])


def test_opening_suggestion_is_soare_with_published_entropy(client):
    r = suggest(client, [], top_k=3)
    assert r.status_code == 200
    body = r.json()
    assert body["remaining"] == 2315 and body["bits_remaining"] == pytest.approx(11.176, abs=1e-3)
    top = body["suggestions"][0]
    assert top["word"] == "soare" and top["entropy"] == pytest.approx(5.886, abs=1e-3)
    assert not top["is_candidate"] and len(body["suggestions"]) == 3


def test_suggest_narrows_candidates_after_feedback(client):
    fb = feedback_to_str(score_guess("soare", "cigar"))
    r = suggest(client, [{"guess": "soare", "feedback": fb}]).json()
    assert 1 < r["remaining"] < 2315 and "cigar" in (r["candidates"] if not r["candidates_truncated"] else ["cigar"])
    assert all(s["entropy"] <= 5.9 for s in r["suggestions"])


def test_full_game_through_the_api_wins_within_six(client):
    rng = random.Random(0)
    answers = client.app.state.wordle.data.answers
    for secret in rng.sample(answers, 8):
        history = []
        for _ in range(6):
            r = suggest(client, history, top_k=1).json()
            guess = r["suggestions"][0]["word"]
            history.append({"guess": guess, "feedback": feedback_to_str(score_guess(guess, secret))})
            if guess == secret:
                break
        assert history[-1]["guess"] == secret, (secret, history)


def test_inconsistent_feedback_reports_zero_remaining_with_message(client):
    r = suggest(client, [{"guess": "crane", "feedback": "ggggg"}, {"guess": "slate", "feedback": "ggggg"}]).json()
    assert r["remaining"] == 0 and r["suggestions"] == [] and "typo" in r["message"]


def test_single_candidate_message(client):
    fb = feedback_to_str(score_guess("crane", "crate"))
    r = suggest(client, [{"guess": "crane", "feedback": fb}, {"guess": "slate", "feedback": feedback_to_str(score_guess("slate", "crate"))}]).json()
    if r["remaining"] == 1:
        assert r["suggestions"][0]["word"] == r["candidates"][0]


def test_hard_mode_suggestions_are_legal(client):
    history = [("crane", feedback_to_str(score_guess("crane", "crate")))]
    r = suggest(client, [{"guess": g, "feedback": f} for g, f in history], hard_mode=True, top_k=15).json()
    parsed = [(g, score_guess(g, "crate")) for g, _ in history]
    assert r["suggestions"]
    assert all(hard_mode_violation(s["word"], parsed) is None for s in r["suggestions"])


def test_minimax_ranker_available(client):
    r = suggest(client, [], solver="minimax", top_k=2).json()
    assert r["suggestions"][0]["word"] == "raise" and r["suggestions"][0]["worst_case"] == 168


@pytest.mark.parametrize("payload, fragment", [
    ({"history": [{"guess": "zzzzz", "feedback": "....."}]}, "allowed guess list"),
    ({"history": [{"guess": "crane", "feedback": "gyxqz"}]}, "feedback must be"),
    ({"history": [{"guess": "cr4ne", "feedback": "....."}]}, "5 letters"),
    ({"history": [], "solver": "oracle"}, "solver must be"),
])
def test_bad_input_is_422_with_a_readable_message(client, payload, fragment):
    r = client.post("/api/wordle/suggest", json=payload)
    assert r.status_code == 422 and fragment in str(r.json()["detail"])


def test_wrong_length_rejected_by_schema(client):
    assert suggest(client, [{"guess": "cranes", "feedback": "....."}]).status_code == 422


def test_simulate_returns_a_consistent_trace(client):
    r = client.post("/api/wordle/simulate", json={"answer": "cigar", "solver": "entropy"}).json()
    assert r["answer"] == "cigar" and r["solved"] and r["guesses"] == len(r["steps"])
    assert r["steps"][-1]["guess"] == "cigar" and r["steps"][-1]["feedback"] == "ggggg"
    first = r["steps"][0]
    assert first["guess"] == "soare" and first["remaining_before"] == 2315
    assert all(s["remaining_after"] <= s["remaining_before"] for s in r["steps"])
    for s in r["steps"]:
        assert s["feedback"] == feedback_to_str(score_guess(s["guess"], "cigar"))


def test_simulate_random_answer_hard_mode_and_other_solvers(client):
    for solver in ("entropy", "minimax", "frequency"):
        r = client.post("/api/wordle/simulate", json={"solver": solver, "hard_mode": True}).json()
        assert r["solved"] and 1 <= r["guesses"] <= 10 and r["hard_mode"] is True


def test_simulate_rejects_unknown_answer(client):
    r = client.post("/api/wordle/simulate", json={"answer": "zzzzz"})
    assert r.status_code == 422 and "possible answers" in r.json()["detail"]


def test_index_and_static_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Word Puzzle Solvers" in page.text
    for path in ("/static/app.js", "/static/style.css"):
        assert client.get(path).status_code == 200
