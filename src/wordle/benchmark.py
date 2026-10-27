"""Benchmark every Wordle approach on all 2,315 original answers.

    python -m src.wordle.benchmark                 # full run, writes results/wordle.{md,json}
    python -m src.wordle.benchmark --limit 100     # quick smoke run (not written unless --out)

Each approach is run in normal and hard mode. Games are capped at MAX_GUESSES (> the
official 6) so average and worst case aren't truncated; "Win ≤6" is the official win rate.
ms/puzzle excludes one-time setup (pattern matrix, opening-move cache), reported separately.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..common.evaluate import summarize, evaluate
from ..common.report import write_report
from .engine import WordleEnv
from .solvers.base import WordleData
from .solvers.entropy import EntropySolver, MinimaxSolver
from .solvers.frequency import FrequencySolver

MAX_GUESSES = 12
THRESHOLDS = (6,)
APPROACHES = {"frequency": FrequencySolver, "entropy": EntropySolver, "minimax": MinimaxSolver}
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--approaches", nargs="+", choices=APPROACHES, default=list(APPROACHES))
    ap.add_argument("--no-hard", action="store_true", help="skip hard-mode runs")
    ap.add_argument("--limit", type=int, help="only the first N answers")
    ap.add_argument("--out", type=Path, help="results dir (default: results/ for full runs only)")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    data = WordleData.load()
    data_seconds = time.perf_counter() - t0
    answers = list(data.answers[: args.limit] if args.limit else data.answers)
    out_dir = args.out or (None if args.limit else RESULTS_DIR)

    summaries, setup_notes = [], []
    for hard in ([False] if args.no_hard else [False, True]):
        for name in args.approaches:
            t0 = time.perf_counter()
            solver = APPROACHES[name](data)
            setup = time.perf_counter() - t0
            env = WordleEnv(data.guesses, max_guesses=MAX_GUESSES, hard_mode=hard)
            label = f"{name} (hard)" if hard else name
            results = evaluate(solver, env, answers)
            s = summarize(label, results, THRESHOLDS)
            summaries.append(s)
            setup_notes.append(f"{label}: {setup:.2f}s")
            print(
                f"{label:<18} win≤6 {s.solved_within[6]:.2%}  avg {s.mean_steps:.3f}  "
                f"worst {s.worst_steps}  {s.ms_per_puzzle_mean:.2f} ms/puzzle",
                file=sys.stderr,
            )

    if out_dir:
        title = f"Wordle — {len(answers)} original answers, {len(data.guesses)} allowed guesses"
        md_path, json_path = write_report(out_dir, "wordle", title, summaries, THRESHOLDS)
        notes = (
            "\n_Games are capped at "
            f"{MAX_GUESSES} guesses; 'Win ≤6' is the official win rate. ms/puzzle excludes "
            f"one-time setup (word lists + pattern matrix loaded from the on-disk cache in {data_seconds:.2f}s; "
            f"a cold build is ~1.8s; then per solver, incl. opening-move search: "
            f"{', '.join(setup_notes)})._\n"
        )
        md_path.write_text(md_path.read_text() + notes)
        print(f"wrote {md_path} and {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
