import numpy as np
import pytest

from src.common.evaluate import evaluate
from src.connections import partition as P
from src.connections.data import DATA_PATH, Group, Puzzle, load_puzzles
from src.connections.env import ConnectionsEnv, ConnectionsState, Proposal, validate_partition
from src.connections.solvers.baselines import EmbeddingModel, EmbeddingSolver, RandomSolver


def make_puzzle() -> Puzzle:
    words = tuple(f"W{i:02d}" for i in range(16))
    groups = tuple(
        Group(f"cat{lv}", lv, frozenset(words[lv * 4 : lv * 4 + 4])) for lv in range(4)
    )
    # board order deliberately shuffled relative to groups
    board = tuple(words[i] for i in (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15))
    return Puzzle(1, "2024-01-01", board, groups)


def perfect(p: Puzzle) -> Proposal:
    return Proposal(tuple(tuple(sorted(g.words)) for g in p.groups))


# --- environment ---------------------------------------------------------------------

def test_env_scores_perfect_and_partial_proposals():
    p, env = make_puzzle(), ConnectionsEnv()
    env.reset(p)
    env.step(perfect(p))
    assert env.done and env.solved and env.metrics()["groups_correct"] == 4.0
    assert all(env.metrics()[f"tier_{t}"] == 1.0 for t in ("yellow", "green", "blue", "purple"))

    # swap one word between the yellow and purple groups: those two wrong, green/blue right
    g = [list(sorted(x.words)) for x in p.groups]
    g[0][0], g[3][0] = g[3][0], g[0][0]
    env.reset(p)
    env.step(Proposal(tuple(tuple(x) for x in g)))
    m = env.metrics()
    assert not env.solved and m["groups_correct"] == 2.0
    assert (m["tier_yellow"], m["tier_green"], m["tier_blue"], m["tier_purple"]) == (0, 1, 1, 0)


def test_env_is_one_shot_and_rejects_invalid_partitions():
    p, env = make_puzzle(), ConnectionsEnv()
    env.reset(p)
    with pytest.raises(ValueError):
        env.step(Proposal((tuple(p.words[:4]),) * 4))  # repeats words
    env.reset(p)
    env.step(perfect(p))
    with pytest.raises(RuntimeError):
        env.step(perfect(p))
    with pytest.raises(ValueError):
        validate_partition([p.words[:5], p.words[5:10], p.words[10:15], p.words[15:]], p.words)


def test_group_order_does_not_matter():
    p, env = make_puzzle(), ConnectionsEnv()
    env.reset(p)
    reordered = Proposal(tuple(reversed([tuple(reversed(sorted(g.words))) for g in p.groups])))
    env.step(reordered)
    assert env.solved


# --- partition space -------------------------------------------------------------------

def test_partition_space_is_complete_and_valid():
    parts = P.all_partitions()
    assert parts.shape == (2_627_625, 4)  # 16! / (4!^4 * 4!)
    cover = P.all_groups()[parts.astype(np.intp)].reshape(len(parts), 16)
    assert (np.sort(cover, axis=1) == np.arange(16)).all()
    assert len(np.unique(np.sort(parts, axis=1), axis=0)) == len(parts)


def test_best_partition_recovers_planted_groups():
    sim = np.zeros((16, 16))
    for k in range(4):
        idx = range(4 * k, 4 * k + 4)
        for i in idx:
            for j in idx:
                sim[i, j] = 1.0
    ((score, parts),) = P.best_partitions(P.pair_sum_scores(sim))
    assert score == pytest.approx(24.0)  # 4 groups x 6 pairs
    assert sorted(map(tuple, parts)) == [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)]


def test_forbidden_group_via_minus_inf_is_never_chosen():
    sim = np.zeros((16, 16))
    for k in range(4):
        for i in range(4 * k, 4 * k + 4):
            for j in range(4 * k, 4 * k + 4):
                sim[i, j] = 1.0
    scores = P.pair_sum_scores(sim)
    banned = P.group_id([0, 1, 2, 3])
    scores[banned] = -np.inf
    ((_, parts),) = P.best_partitions(scores)
    assert [0, 1, 2, 3] not in parts


def test_top_k_is_sorted_and_distinct():
    scores = np.random.default_rng(0).random(1820)
    top = P.best_partitions(scores, top_k=5)
    totals = [t for t, _ in top]
    assert totals == sorted(totals, reverse=True)
    assert len({tuple(map(tuple, p)) for _, p in top}) == 5


# --- solvers (fake embedder: no model download in unit tests) ---------------------------

class FakeEmbedder(EmbeddingModel):
    """Words in the same planted group get nearly identical vectors."""

    def __init__(self, group_of):
        super().__init__("fake")
        self.group_of = group_of

    def vectors(self, words):
        rng = np.random.default_rng(0)
        centers = rng.normal(size=(4, 8))
        out = []
        for w in words:
            v = centers[self.group_of[w]] + 0.01 * rng.normal(size=8)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def test_embedding_solver_solves_cleanly_separable_puzzle():
    p = make_puzzle()
    group_of = {w: g.level for g in p.groups for w in g.words}
    solver = EmbeddingSolver(embedder=FakeEmbedder(group_of))
    (result,) = evaluate(solver, ConnectionsEnv(), [p])
    assert result.solved and result.metrics["groups_correct"] == 4.0


def test_embedding_solver_returns_confidences_per_group():
    p = make_puzzle()
    solver = EmbeddingSolver(embedder=FakeEmbedder({w: g.level for g in p.groups for w in g.words}))
    prop = solver.next_action(ConnectionsState(p.words))
    assert len(prop.confidences) == 4 and all(c > 0.9 for c in prop.confidences)


def test_random_solver_always_valid_and_seeded():
    p = make_puzzle()
    a, b = RandomSolver(seed=7), RandomSolver(seed=7)
    for _ in range(20):
        pa = a.next_action(ConnectionsState(p.words))
        validate_partition(pa.groups, p.words)
        assert pa == b.next_action(ConnectionsState(p.words))


# --- real dataset ----------------------------------------------------------------------

@pytest.mark.skipif(not DATA_PATH.exists(), reason="run data/download_connections.py")
def test_dataset_loads_and_is_well_formed():
    puzzles = load_puzzles()
    assert len(puzzles) == 628
    assert [p.date for p in puzzles] == sorted(p.date for p in puzzles)
    for p in puzzles:
        assert len(set(p.words)) == 16
        assert sorted(g.level for g in p.groups) == [0, 1, 2, 3]
        assert set().union(*(g.words for g in p.groups)) == set(p.words)
    first = next(p for p in puzzles if p.id == 1)
    assert first.date == "2023-06-12"
    assert len(first.words) == 16 and all(g.label for g in first.groups)
