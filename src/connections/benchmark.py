"""Benchmark Connections approaches on the dev dataset.

    python -m src.connections.benchmark --approaches random minilm mpnet   # no API key needed
    python -m src.connections.benchmark --approaches haiku-llm sonnet-hybrid \
        --sample 150 --probe                                               # needs ANTHROPIC_API_KEY
    python -m src.connections.benchmark --report-only                      # rebuild tables

Per-episode results are saved to results/episodes/ (puzzle ids and numbers only, no NYT
text) so slices can be rebuilt without re-running models. Every approach in one report is
scored on the SAME puzzle subset.

LEAKAGE: all dev puzzles predate the models' training cutoffs, so LLM scores are an UPPER
BOUND on performance on genuinely new puzzles. The memorization probe (see probe.py) flags
puzzles the model can recite from a date alone; the report splits results by that flag and by
date. A clean post-cutoff holdout does not exist in this project.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

from ..common.evaluate import EpisodeResult, evaluate, summarize
from ..common.report import markdown_report
from ..common.solver import Usage
from .data import TIERS, Puzzle, load_puzzles
from .env import ConnectionsEnv
from .probe import ProbeResult, probe_summary, run_probe
from .solvers.baselines import EmbeddingModel, EmbeddingSolver, RandomSolver

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
EPISODES_DIR = RESULTS_DIR / "episodes"
METRIC_COLS = [("groups_correct", "Avg groups (of 4)", "num")] + [
    (f"tier_{t}", t.capitalize(), "pct") for t in TIERS
]
MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
EMBEDDERS: dict[str, EmbeddingModel] = {}


def embedder(name: str) -> EmbeddingModel:
    return EMBEDDERS.setdefault(name, EmbeddingModel(name))


def make_solver(approach: str, client):
    """Approach names: random | minilm | mpnet | {haiku,sonnet}-{llm,hybrid,hybrid-noemb}."""
    if approach == "random":
        return RandomSolver(seed=0)
    if approach == "minilm":
        return EmbeddingSolver("all-MiniLM-L6-v2", embedder("all-MiniLM-L6-v2"))
    if approach == "mpnet":
        return EmbeddingSolver("all-mpnet-base-v2", embedder("all-mpnet-base-v2"))
    model_key, _, kind = approach.partition("-")
    from .solvers.hybrid import HybridSolver
    from .solvers.llm import LLMSolver

    model = MODELS[model_key]
    if kind == "llm":
        return LLMSolver(client, model)
    if kind == "hybrid":
        return HybridSolver(client, model, embedder=embedder("all-mpnet-base-v2"))
    if kind == "hybrid-noemb":
        return HybridSolver(client, model, alpha=0.0)
    raise ValueError(f"unknown approach {approach!r}")


def is_llm(approach: str) -> bool:
    return approach.split("-")[0] in MODELS


# --- persistence ---------------------------------------------------------------------

def save_episodes(approach: str, name: str, results: list[EpisodeResult]) -> None:
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "episodes": [asdict(r) for r in results]}
    (EPISODES_DIR / f"connections_{approach}.json").write_text(json.dumps(payload, indent=1))


def load_episodes(approach: str) -> tuple[str, dict[str, EpisodeResult]] | None:
    path = EPISODES_DIR / f"connections_{approach}.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    eps = {e["puzzle_id"]: EpisodeResult(**{**e, "usage": Usage(**e["usage"])}) for e in raw["episodes"]}
    return raw["name"], eps


def save_probe(model_key: str, results: list[ProbeResult]) -> None:
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    (EPISODES_DIR / f"probe_{model_key}.json").write_text(
        json.dumps([asdict(r) for r in results], indent=1))


def load_probe(model_key: str) -> dict[str, ProbeResult] | None:
    path = EPISODES_DIR / f"probe_{model_key}.json"
    if not path.exists():
        return None
    return {r["puzzle_id"]: ProbeResult(**r) for r in json.loads(path.read_text())}


# --- reporting -----------------------------------------------------------------------

def choose_subset(puzzles: list[Puzzle], sample: int | None, seed: int, limit: int | None) -> list[Puzzle]:
    if limit:
        return puzzles[:limit]
    if sample and sample < len(puzzles):
        rng = random.Random(seed)
        return sorted(rng.sample(puzzles, sample), key=lambda p: (p.date, p.id))
    return puzzles


def build_report(subset: list[Puzzle], approaches: list[str]) -> tuple[str, dict]:
    ids = [str(p.id) for p in subset]
    loaded = {a: load_episodes(a) for a in approaches}
    loaded = {a: v for a, v in loaded.items() if v and all(i in v[1] for i in ids)}
    third = max(1, len(subset) // 3)
    slices = {
        f"All {len(subset)} puzzles": ids,
        f"Most recent third ({subset[-third].date} to {subset[-1].date}, n={third})": ids[-third:],
    }
    sections, data = [], {}
    for title, sl in slices.items():
        sums = [summarize(loaded[a][0], [loaded[a][1][i] for i in sl]) for a in loaded]
        show_cost = any(is_llm(a) for a in loaded)
        sections.append(markdown_report(title, sums, solve_label="Fully solved", metric_cols=METRIC_COLS,
                                        show_steps=False, show_histograms=False, show_cost=show_cost))
        data[title] = [s.to_dict() for s in sums]

    probe_lines = []
    for mk in MODELS:
        probe = load_probe(mk)
        llm_apps = [a for a in loaded if a.startswith(mk + "-")]
        if not probe or not llm_apps:
            continue
        pr = [probe[i] for i in ids if i in probe]
        ps = probe_summary(pr)
        probe_lines.append(
            f"- **{MODELS[mk]}** recalls {ps['mean_labels_matched']:.2f} of 4 category names from a "
            f"puzzle's number and date alone (chance control: {ps['mean_labels_matched_control']:.2f}); "
            f"{ps['flagged_rate']:.0%} of puzzles flagged likely-seen (>=2 of 4 matched).")
        rows = []
        for a in llm_apps:
            for flag, label in ((True, "flagged (likely seen)"), (False, "not flagged")):
                sl = [i for i in ids if i in probe and probe[i].flagged == flag]
                if sl:
                    rows.append(summarize(f"{loaded[a][0]} - {label}", [loaded[a][1][i] for i in sl]))
        if rows:
            sections.append(markdown_report(f"Split by memorization flag ({MODELS[mk]} probe)", rows,
                                            solve_label="Fully solved", metric_cols=METRIC_COLS,
                                            show_steps=False, show_histograms=False))
            data[f"flag split {mk}"] = [s.to_dict() for s in rows]
    header = [
        "> **Upper bound.** Every puzzle here predates the models' training cutoffs, so LLM scores "
        "are an upper bound on performance on genuinely new puzzles. No clean post-cutoff holdout "
        "exists in this project.",
        "",
    ]
    if probe_lines:
        header += ["### Memorization probe", *probe_lines, ""]
    return "\n".join(header + sections), data


# --- main ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--approaches", nargs="+", default=["random", "minilm", "mpnet"])
    ap.add_argument("--sample", type=int, help="seeded random subset of N puzzles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="first N puzzles by date (smoke test)")
    ap.add_argument("--probe", action="store_true", help="run the memorization probe for LLM models used")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out-name", default="connections", help="results/<name>.md|json")
    args = ap.parse_args(argv)

    subset = choose_subset(load_puzzles(), args.sample, args.seed, args.limit)
    print(f"{len(subset)} puzzles", file=sys.stderr)

    if not args.report_only:
        client = None
        if any(is_llm(a) for a in args.approaches) or args.probe:
            from ..common.llm import AnthropicJSONClient
            client = AnthropicJSONClient()
        for approach in args.approaches:
            solver = make_solver(approach, client)
            results = evaluate(solver, ConnectionsEnv(), subset)
            save_episodes(approach, solver.name, results)
            s = summarize(solver.name, results)
            print(f"{solver.name:<40} solved {s.solve_rate:.1%}  groups {s.mean_metrics['groups_correct']:.2f}/4"
                  f"  ${s.cost_usd_per_puzzle:.4f}/puzzle", file=sys.stderr)
        if args.probe:
            for mk in sorted({a.split('-')[0] for a in args.approaches if is_llm(a)}):
                save_probe(mk, run_probe(client, MODELS[mk], subset))
                print(f"probe done: {mk}", file=sys.stderr)

    if not args.limit:
        order = ["random", "minilm", "mpnet"] + [f"{m}-{k}" for m in MODELS for k in ("llm", "hybrid-noemb", "hybrid")]
        md, data = build_report(subset, order)
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / f"{args.out_name}.md").write_text(md)
        (RESULTS_DIR / f"{args.out_name}.json").write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote results/{args.out_name}.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
