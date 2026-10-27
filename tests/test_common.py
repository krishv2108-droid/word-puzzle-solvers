import json

import pytest

from src.common.evaluate import evaluate, summarize
from src.common.report import markdown_report, write_report
from src.common.solver import Solver
from src.wordle.engine import WordleEnv, WordleState

WORDS = ["crane", "slate", "irate", "trace", "crate", "grace", "apple", "abide"]


class ScriptedSolver(Solver[WordleState, str]):
    """Guesses a fixed sequence; stands in for a real solver and fakes LLM usage."""

    name = "scripted"

    def __init__(self, script):
        super().__init__()
        self.script = script

    def next_action(self, state):
        self.usage.input_tokens += 100
        self.usage.output_tokens += 10
        self.usage.cost_usd += 0.01
        return self.script[state.guesses_made]


def test_runner_collects_per_episode_results():
    env = WordleEnv(WORDS, max_guesses=4)
    solver = ScriptedSolver(["slate", "irate", "crate", "grace"])
    results = evaluate(solver, env, ["crate", "slate", "apple"])
    assert [(r.puzzle_id, r.solved, r.steps) for r in results] == [
        ("crate", True, 3),
        ("slate", True, 1),
        ("apple", False, 4),
    ]
    # usage is reported per episode (a delta), not cumulative
    assert [r.usage.api_calls for r in results] == [0, 0, 0]
    assert [round(r.usage.cost_usd, 2) for r in results] == [0.03, 0.01, 0.04]
    assert all(r.seconds >= 0 for r in results)


def test_summary_statistics():
    env = WordleEnv(WORDS, max_guesses=4)
    solver = ScriptedSolver(["slate", "irate", "crate", "grace"])
    results = evaluate(solver, env, ["crate", "slate", "apple", "irate"])
    s = summarize("scripted", results, thresholds=(1, 3))
    assert s.n == 4 and s.unsolved == 1
    assert s.solve_rate == 0.75
    assert s.solved_within == {1: 0.25, 3: 0.75}
    assert s.histogram == {1: 1, 2: 1, 3: 1}
    assert s.worst_steps == 4
    assert s.mean_steps == pytest.approx((3 + 1 + 4 + 2) / 4)
    assert s.mean_steps_solved == pytest.approx(2.0)
    assert s.cost_usd_total == pytest.approx(0.10)
    assert s.input_tokens_per_puzzle == pytest.approx(250)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize("x", [])


def test_report_outputs_markdown_and_json(tmp_path):
    env = WordleEnv(WORDS, max_guesses=4)
    solver = ScriptedSolver(["slate", "irate", "crate", "grace"])
    s = summarize("scripted", evaluate(solver, env, ["crate", "slate"]), thresholds=(6,))
    md = markdown_report("Wordle", [s], thresholds=(6,), show_cost=True)
    assert "| Approach |" in md and "scripted" in md and "Win ≤6" in md and "$/puzzle" in md
    md_path, json_path = write_report(tmp_path, "wordle", "Wordle", [s], thresholds=(6,))
    data = json.loads(json_path.read_text())
    assert data[0]["name"] == "scripted" and data[0]["histogram"] == {"1": 1, "3": 1}
    assert md_path.read_text().startswith("## Wordle")
