"""Learned-scorer experiments with strict dev/test discipline.

    python -m src.connections.experiments             # ablations by 5-fold CV on DEV only
    python -m src.connections.experiments --final     # train on all DEV, score the TEST sample ONCE

All choices (features, hyperparameters, fusion weights) are made on DEV (the 478 puzzles not in
the 150-puzzle test sample). `--final` is the only code path that touches TEST, and it takes no
tuning knobs, so running it more than once cannot change what it reports.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ..common.evaluate import evaluate, summarize
from .benchmark import EPISODES_DIR, split_test_dev
from .data import load_puzzles
from .env import ConnectionsEnv
from .learned import (LearnedModel, LearnedSolver, cross_validate, feature_matrix, fit_pair_model,
                      groups_correct, solve_from_pair_scores)
from .solvers.baselines import EmbeddingModel

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MPNET, MINILM, BGE = "all-mpnet-base-v2", "all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5"


def embedders() -> dict[str, EmbeddingModel]:
    return {"mpnet": EmbeddingModel(MPNET), "minilm": EmbeddingModel(MINILM), "bge": EmbeddingModel(BGE)}


def ci(vals) -> tuple[float, float]:
    from .benchmark import bootstrap_ci

    return bootstrap_ci(vals)


def configs(names: list[str]) -> dict[str, list[int]]:
    pick = lambda pred: [i for i, n in enumerate(names) if pred(n)]  # noqa: E731
    return {
        "mpnet cosine only (trained)": pick(lambda n: n == "mpnet_cos"),
        "mpnet embedding features": pick(lambda n: n.startswith("mpnet_")),
        "mpnet + MiniLM embedding features": pick(lambda n: n.startswith(("mpnet_", "minilm_"))),
        "bge-large embedding features": pick(lambda n: n.startswith("bge_")),
        "mpnet + MiniLM + bge-large embedding features": pick(lambda n: n.startswith(("mpnet_", "minilm_", "bge_"))),
        "string features only": pick(lambda n: n.startswith("str_")),
        "all features (3 embedding models + string)": list(range(len(names))),
    }


def run_dev_ablation() -> str:
    puzzles = load_puzzles()
    _, dev = split_test_dev(puzzles)
    t0 = time.perf_counter()
    X, y, names = feature_matrix(dev, embedders())
    print(f"features {X.shape} in {time.perf_counter() - t0:.0f}s", file=sys.stderr)

    # untrained baseline in the same harness: pair score = raw mpnet cosine
    cos_col = names.index("mpnet_cos")
    base = np.array([groups_correct(p, solve_from_pair_scores(X[i, :, cos_col])[0]) for i, p in enumerate(dev)])
    rows = [("mpnet cosine, untrained (current baseline)", base)]
    for label, cols in configs(names).items():
        t0 = time.perf_counter()
        rows.append((label, cross_validate(dev, X, y, cols)))
        print(f"{label}: {rows[-1][1].mean():.3f} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)

    lines = [f"## Learned pair scorer: ablation on DEV (n={len(dev)}, 5-fold CV by puzzle)", "",
             "| Features | Avg groups (of 4) | 95% CI | Fully solved | Paired gain vs baseline (95% CI) |",
             "|---|---|---|---|---|"]
    for label, c in rows:
        lo, hi = ci(c)
        d_lo, d_hi = ci(c - base)
        gain = "n/a" if c is base else f"{(c - base).mean():+.2f} ({d_lo:+.2f} to {d_hi:+.2f})"
        lines.append(f"| {label} | {c.mean():.2f} | {lo:.2f} to {hi:.2f} | {(c == 4).mean():.1%} | {gain} |")
    return "\n".join(lines) + "\n"


def run_final(bonus: float | None = None) -> None:
    puzzles = load_puzzles()
    test, dev = split_test_dev(puzzles)
    embs = embedders()
    X, y, names = feature_matrix(dev, embs)
    cols = list(range(len(names)))  # fixed a priori: all features; not chosen from TEST
    model = fit_pair_model(X[:, :, cols], y)
    solver = LearnedSolver(LearnedModel(model, embs, names, cols), "learned pair scorer")
    results = evaluate(solver, ConnectionsEnv(), test)
    s = summarize(solver.name, results)
    print(f"TEST n={s.n}: solved {s.solve_rate:.1%} groups {s.mean_metrics['groups_correct']:.3f}/4 "
          f"tiers {[round(s.mean_metrics[f'tier_{t}'], 3) for t in ('yellow','green','blue','purple')]}")
    from .benchmark import save_episodes
    save_episodes("learned", solver.name, results)

    if bonus is not None:
        from ..common.local_llm import LocalJSONClient
        from .fusion import FusedSolver

        fused = FusedSolver(LocalJSONClient(), LOCAL_MODEL, solver.learned, bonus,
                            name="learned + LLM (qwen3-8b local)")
        results = evaluate(fused, ConnectionsEnv(), test)
        s = summarize(fused.name, results)
        print(f"TEST n={s.n} fused (bonus={bonus}): solved {s.solve_rate:.1%} "
              f"groups {s.mean_metrics['groups_correct']:.3f}/4 llm_failures {s.mean_metrics.get('llm_failure', 0):.3f}")
        save_episodes("learned-llm", fused.name, results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--bonus", type=float, help="with --final: also score learned+LLM fusion at this bonus")
    ap.add_argument("--fusion-dev", action="store_true", help="sweep the fusion bonus on DEV")
    args = ap.parse_args()
    if args.fusion_dev:
        md, best = run_fusion_dev()
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / "connections_fusion_dev.md").write_text(md)
        print(md)
    elif args.final:
        run_final(args.bonus)
    else:
        md = run_dev_ablation()
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / "connections_learned_dev.md").write_text(md)
        print(md)


# --- fusion with LLM candidates (tuned on DEV only) -------------------------------------------

BONUS_GRID = [0, 0.5, 1, 2, 4, 8, 16, 32]
LOCAL_MODEL = "mlx-community/Qwen3-8B-4bit"


def oof_logits(dev, X, y, cols, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold pair logits for every DEV puzzle (the scorer never saw that puzzle)."""
    from .learned import pair_logits

    fold_of = np.random.default_rng(seed).permutation(len(dev)) % folds
    out = np.zeros(y.shape)
    for k in range(folds):
        tr, te = fold_of != k, fold_of == k
        out[te] = pair_logits(fit_pair_model(X[tr][:, :, cols], y[tr], seed), X[te][:, :, cols])
    return out


def cached_candidates(puzzles) -> dict[int, list[dict]]:
    """LLM candidate groups for each puzzle, read from the response cache (never generates)."""
    from ..common.local_llm import LocalJSONClient
    from .solvers.hybrid import CANDIDATES_SCHEMA, SYSTEM_PROMPT
    from .solvers.llm import user_prompt

    client = LocalJSONClient()
    out = {}
    for p in puzzles:
        req = dict(backend="local-mlx-thinking=False", model=LOCAL_MODEL, system=SYSTEM_PROMPT,
                   user=user_prompt(p.words), schema=CANDIDATES_SCHEMA, max_new_tokens=client.max_new_tokens)
        hit = client.cache.get(req)
        if hit is not None:
            out[p.id] = (hit.data or {}).get("candidates", [])
    return out


def run_fusion_dev(n: int = 100, seed: int = 1) -> str:
    from .benchmark import choose_subset
    from .fusion import fused_partition, llm_group_scores

    _, dev = split_test_dev(load_puzzles())
    embs = embedders()
    X, y, names = feature_matrix(dev, embs)
    cols = list(range(len(names)))
    logits = oof_logits(dev, X, y, cols)
    pos = {p.id: i for i, p in enumerate(dev)}
    chosen = choose_subset(dev, n, seed, None)
    cands = cached_candidates(chosen)
    missing = [p.id for p in chosen if p.id not in cands]
    if missing:
        raise SystemExit(f"{len(missing)} dev puzzles have no cached LLM candidates yet")
    rows = []
    for b in BONUS_GRID:
        c = np.array([groups_correct(p, fused_partition(logits[pos[p.id]], llm_group_scores(p.words, cands[p.id])[0], b)[0])
                      for p in chosen])
        rows.append((b, c))
    base = rows[0][1]
    lines = [f"## Fusion weight sweep on DEV (n={len(chosen)} puzzles with LLM candidates, out-of-fold scorer)", "",
             "| LLM bonus (log-odds per unit confidence) | Avg groups (of 4) | 95% CI | Fully solved | Gain vs learned alone |",
             "|---|---|---|---|---|"]
    for b, c in rows:
        lo, hi = ci(c)
        g_lo, g_hi = ci(c - base)
        lines.append(f"| {b} | {c.mean():.2f} | {lo:.2f} to {hi:.2f} | {(c == 4).mean():.1%} | "
                     f"{(c - base).mean():+.2f} ({g_lo:+.2f} to {g_hi:+.2f}) |")
    best = max(rows, key=lambda r: (r[1].mean(), -r[0]))[0]
    lines += ["", f"Selected bonus (best dev mean; ties to the smaller): **{best}**"]
    return "\n".join(lines) + "\n", best
