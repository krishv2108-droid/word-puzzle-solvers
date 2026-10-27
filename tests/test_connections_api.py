import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.connections_service import ConnectionsService
from src.api.main import app
from src.connections.calibration import Calibrator
from src.connections.learned import LearnedModel, puzzle_features
from src.connections.solvers.baselines import EmbeddingModel

WORDS = [f"{c}{i}" for c in "ABCD" for i in range(1, 5)]  # planted groups: same letter
BOARD = [WORDS[i] for i in (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)]
TRUE_GROUPS = [{w for w in WORDS if w[0] == c} for c in "ABCD"]


class FakeEmbedder(EmbeddingModel):
    def __init__(self):
        super().__init__("fake", cache_dir=None)

    def vectors(self, words):
        centers = np.random.default_rng(0).normal(size=(4, 12))
        rng = np.random.default_rng(1)
        out = []
        for w in words:
            k = "ABCD".find(w[0].upper())
            v = (centers[k] if k >= 0 else rng.normal(size=12)) + 0.05 * rng.normal(size=12)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


class FakeModel:
    """Same-group probability rises with the mpnet cosine feature."""

    def __init__(self, col):
        self.col = col

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-8 * (X[:, self.col] - 0.5)))
        return np.stack([1 - p, p], axis=1)


def make_service():
    embs = {t: FakeEmbedder() for t in ("mpnet", "minilm", "bge")}
    names = list(puzzle_features(tuple(BOARD), embs))
    learned = LearnedModel(FakeModel(names.index("mpnet_cos")), embs, names, list(range(len(names))))
    cal = Calibrator().fit(np.array([[0.2, 0.5], [0.5, 2.0], [1.5, 6.0], [3.0, 15.0], [0.1, 0.1], [2.0, 9.0]]),
                           np.array([0, 0, 1, 1, 0, 1]))
    return ConnectionsService(learned, cal)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        c.app.state.connections = make_service()
        yield c
        c.app.state.connections = None


def solve(client, words=BOARD):
    return client.post("/api/connections/solve", json={"words": words})


def groups_of(resp_groups):
    return {frozenset(g["words"]) for g in resp_groups}


def test_solve_recovers_planted_groups_with_sorted_confidences(client):
    r = solve(client).json()
    assert groups_of(r["groups"]) == {frozenset(g) for g in TRUE_GROUPS}
    conf = [g["confidence"] for g in r["groups"]]
    assert conf == sorted(conf, reverse=True) and all(0 <= c <= 1 for c in conf)
    assert r["expected_correct"] == pytest.approx(sum(conf))


def test_solve_is_case_and_whitespace_insensitive(client):
    messy = [("  " + w.lower() + " ") for w in BOARD]
    r = solve(client, messy).json()
    assert groups_of(r["groups"]) == {frozenset(g) for g in TRUE_GROUPS}


@pytest.mark.parametrize("words, fragment", [
    (BOARD[:15], "exactly 16"),
    (BOARD + ["X9"], "exactly 16"),
    (BOARD[:15] + [BOARD[0].lower()], "different"),
    (BOARD[:15] + ["  "], "needs a word"),
    (BOARD[:15] + ["x" * 41], "at most"),
])
def test_solve_validates_input_with_readable_422(client, words, fragment):
    r = solve(client, words)
    assert r.status_code == 422 and fragment in r.json()["detail"]


def test_hint_returns_one_true_group_then_respects_solved_and_wrong(client):
    h = client.post("/api/connections/hint", json={"words": BOARD}).json()
    first = frozenset(h["group"]["words"])
    assert first in {frozenset(g) for g in TRUE_GROUPS} and h["remaining_groups"] == 4

    h2 = client.post("/api/connections/hint", json={"words": BOARD, "solved": [sorted(first)]}).json()
    second = frozenset(h2["group"]["words"])
    assert second != first and not (second & first) and h2["remaining_groups"] == 3

    wrong = sorted(first)
    h3 = client.post("/api/connections/hint", json={"words": BOARD, "wrong": [wrong]}).json()
    assert frozenset(h3["group"]["words"]) != first


def test_hint_all_solved_message(client):
    solved = [sorted(g) for g in TRUE_GROUPS]
    r = client.post("/api/connections/hint", json={"words": BOARD, "solved": solved}).json()
    assert r["group"] is None and "already solved" in r["message"]


@pytest.mark.parametrize("payload, fragment", [
    ({"solved": [["A1", "A2", "A3", "NOPE"]]}, "not on the board"),
    ({"solved": [["A1", "A2", "A3"]]}, "4 different"),
    ({"solved": [["A1", "A2", "A3", "A4"], ["A1", "B1", "B2", "B3"]]}, "must not overlap"),
])
def test_hint_validates_group_arguments(client, payload, fragment):
    r = client.post("/api/connections/hint", json={"words": BOARD, **payload})
    assert r.status_code == 422 and fragment in r.json()["detail"]


def test_hint_with_impossible_constraints_is_a_readable_422(client):
    # three groups solved forces the fourth; marking that forced group wrong leaves no partition
    solved = [sorted(g) for g in TRUE_GROUPS[:3]]
    r = client.post("/api/connections/hint",
                    json={"words": BOARD, "solved": solved, "wrong": [sorted(TRUE_GROUPS[3])]})
    assert r.status_code == 422 and "no valid partition" in r.json()["detail"]


def test_oversized_wrong_list_is_rejected_by_the_schema(client):
    r = client.post("/api/connections/hint", json={"words": BOARD, "wrong": [BOARD[:4]] * 41})
    assert r.status_code == 422


def test_connections_unavailable_is_503_with_instructions():
    with TestClient(app) as c:
        c.app.state.connections = None
        assert c.get("/api/connections/info").json() == {"available": False}
        r = c.post("/api/connections/solve", json={"words": BOARD})
        assert r.status_code == 503 and "src.connections.train" in r.json()["detail"]
