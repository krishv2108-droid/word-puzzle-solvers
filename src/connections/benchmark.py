"""Benchmark Connections approaches on the dev dataset.

    python -m src.connections.benchmark                 # writes results/connections.{md,json}
    python -m src.connections.benchmark --limit 40      # smoke run

Metrics: fully-solved rate, average correct groups (0-4), accuracy per difficulty tier, and
latency (plus cost for LLM approaches). Every approach is reported on all puzzles and on the
most recent RECENT puzzles. For non-LLM approaches the split is only a consistency check; it
matters for LLMs, whose training data may contain older puzzles (see README: data leakage).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..common.evaluate import EpisodeResult, evaluate, summarize
from ..common.report import markdown_report
from .data import TIERS, Puzzle, load_puzzles
from .env import ConnectionsEnv
from .solvers.baselines import EmbeddingModel, EmbeddingSolver, RandomSolver

RECENT = 100
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
METRIC_COLS = [("groups_correct", "Avg groups (of 4)", "num")] + [
    (f"tier_{t}", t.capitalize(), "pct") for t in TIERS
]


def build_solvers(names: list[str]):
    makers = {
        "random": lambda: RandomSolver(seed=0),
        "minilm": lambda: EmbeddingSolver("all-MiniLM-L6-v2"),
        "mpnet": lambda: EmbeddingSolver("all-mpnet-base-v2"),
    }
    return [makers[n]() for n in names]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--approaches", nargs="+", default=["random", "minilm", "mpnet"])
    ap.add_argument("--limit", type=int, help="only the first N puzzles (oldest first)")
    ap.add_argument("--out", type=Path, help="results dir (default results/ unless --limit)")
    args = ap.parse_args(argv)

    puzzles = load_puzzles()
    if args.limit:
        puzzles = puzzles[: args.limit]
    out_dir = args.out or (None if args.limit else RESULTS_DIR)
    slices: dict[str, list[Puzzle]] = {f"All {len(puzzles)} puzzles": puzzles}
    if len(puzzles) > RECENT:
        recent = puzzles[-RECENT:]
        slices[f"Most recent {RECENT} puzzles ({recent[0].date} to {recent[-1].date})"] = recent

    all_results: dict[str, dict[str, list[EpisodeResult]]] = {s: {} for s in slices}
    for solver in build_solvers(args.approaches):
        env = ConnectionsEnv()
        results = evaluate(solver, env, puzzles)
        by_id = {r.puzzle_id: r for r in results}
        for slice_name, subset in slices.items():
            all_results[slice_name][solver.name] = [by_id[str(p.id)] for p in subset]
        s = summarize(solver.name, results)
        print(f"{solver.name:<38} solved {s.solve_rate:.2%}  groups {s.mean_metrics['groups_correct']:.2f}/4",
              file=sys.stderr)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        md, js = [], {}
        for slice_name, per_solver in all_results.items():
            sums = [summarize(n, r) for n, r in per_solver.items()]
            md.append(markdown_report(slice_name, sums, solve_label="Fully solved",
                                      metric_cols=METRIC_COLS, show_steps=False, show_histograms=False))
            js[slice_name] = [s.to_dict() for s in sums]
        (out_dir / "connections.md").write_text("\n".join(md))
        (out_dir / "connections.json").write_text(json.dumps(js, indent=2) + "\n")
        print(f"wrote {out_dir / 'connections.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
