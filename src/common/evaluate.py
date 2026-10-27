"""Shared evaluation runner: play every puzzle with a solver and aggregate metrics."""
from __future__ import annotations

import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Sequence

from .solver import Environment, Solver, Usage


@dataclass
class EpisodeResult:
    puzzle_id: str
    solved: bool
    steps: int
    seconds: float
    usage: Usage
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class Summary:
    name: str
    n: int
    solve_rate: float
    solved_within: dict[int, float]  # step threshold -> fraction solved in <= threshold
    mean_steps: float  # over all episodes; unsolved episodes count their steps used
    mean_steps_solved: float
    worst_steps: int
    histogram: dict[int, int]  # steps -> count (solved episodes only)
    unsolved: int
    ms_per_puzzle_mean: float
    ms_per_puzzle_p95: float
    ms_per_puzzle_max: float
    cost_usd_total: float
    cost_usd_per_puzzle: float
    input_tokens_per_puzzle: float
    output_tokens_per_puzzle: float
    mean_metrics: dict[str, float]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["histogram"] = {str(k): v for k, v in sorted(self.histogram.items())}
        d["solved_within"] = {str(k): v for k, v in sorted(self.solved_within.items())}
        return d


def run_episode(env: Environment, solver: Solver, puzzle) -> EpisodeResult:
    solver.reset()
    before = solver.usage.snapshot()
    start = time.perf_counter()
    state = env.reset(puzzle)
    while not env.done:
        state = env.step(solver.next_action(state))
    usage = solver.usage.minus(before)
    # A cache hit returns instantly; add back the latency it originally took so that timing
    # is the same whether or not a reply was replayed.
    seconds = time.perf_counter() - start + usage.cached_latency_s
    return EpisodeResult(
        puzzle_id=env.puzzle_id,
        solved=env.solved,
        steps=env.steps,
        seconds=seconds,
        usage=usage,
        metrics={**env.metrics(), **solver.metrics()},
    )


def evaluate(
    solver: Solver,
    env: Environment,
    puzzles: Iterable,
    *,
    on_result: Callable[[EpisodeResult], None] | None = None,
) -> list[EpisodeResult]:
    results = []
    for puzzle in puzzles:
        result = run_episode(env, solver, puzzle)
        results.append(result)
        if on_result:
            on_result(result)
    return results


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(
    name: str, results: Sequence[EpisodeResult], thresholds: Sequence[int] = ()
) -> Summary:
    if not results:
        raise ValueError("no results to summarize")
    n = len(results)
    solved = [r for r in results if r.solved]
    ms = [r.seconds * 1000 for r in results]
    metric_keys = sorted({k for r in results for k in r.metrics})
    return Summary(
        name=name,
        n=n,
        solve_rate=len(solved) / n,
        solved_within={t: sum(r.steps <= t for r in solved) / n for t in thresholds},
        mean_steps=statistics.fmean(r.steps for r in results),
        mean_steps_solved=statistics.fmean(r.steps for r in solved) if solved else float("nan"),
        worst_steps=max(r.steps for r in results),
        histogram=dict(Counter(r.steps for r in solved)),
        unsolved=n - len(solved),
        ms_per_puzzle_mean=statistics.fmean(ms),
        ms_per_puzzle_p95=_percentile(ms, 0.95),
        ms_per_puzzle_max=max(ms),
        cost_usd_total=sum(r.usage.cost_usd for r in results),
        cost_usd_per_puzzle=sum(r.usage.cost_usd for r in results) / n,
        input_tokens_per_puzzle=sum(r.usage.input_tokens for r in results) / n,
        output_tokens_per_puzzle=sum(r.usage.output_tokens for r in results) / n,
        mean_metrics={
            k: statistics.fmean(r.metrics[k] for r in results if k in r.metrics)
            for k in metric_keys
        },
    )
