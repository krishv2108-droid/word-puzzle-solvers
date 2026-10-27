"""Render benchmark summaries as a markdown table plus machine-readable JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .evaluate import Summary


def _histogram_bars(hist: dict[int, int], total: int, width: int = 24) -> list[str]:
    if not hist:
        return []
    peak = max(hist.values())
    return [
        f"{k:>2} | {'█' * max(1, round(width * v / peak))} {v} ({v / total:.1%})"
        for k, v in sorted(hist.items())
    ]


def markdown_table(
    summaries: Sequence[Summary], thresholds: Sequence[int] = (), show_cost: bool = False
) -> str:
    headers = ["Approach", "N", "Solve rate"]
    headers += [f"Win ≤{t}" for t in thresholds]
    headers += ["Avg steps", "Worst", "ms/puzzle (mean / p95)"]
    if show_cost:
        headers += ["$/puzzle", "In tok", "Out tok"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for s in summaries:
        row = [s.name, str(s.n), f"{s.solve_rate:.2%}"]
        row += [f"{s.solved_within[t]:.2%}" for t in thresholds]
        row += [
            f"{s.mean_steps:.3f}",
            str(s.worst_steps),
            f"{s.ms_per_puzzle_mean:.2f} / {s.ms_per_puzzle_p95:.2f}",
        ]
        if show_cost:
            row += [
                f"${s.cost_usd_per_puzzle:.4f}",
                f"{s.input_tokens_per_puzzle:.0f}",
                f"{s.output_tokens_per_puzzle:.0f}",
            ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def markdown_report(
    title: str,
    summaries: Sequence[Summary],
    thresholds: Sequence[int] = (),
    show_cost: bool = False,
) -> str:
    parts = [f"## {title}", "", markdown_table(summaries, thresholds, show_cost), ""]
    for s in summaries:
        parts += [f"**{s.name}** — guess-count distribution", "```"]
        parts += _histogram_bars(s.histogram, s.n)
        if s.unsolved:
            parts.append(f"unsolved: {s.unsolved} ({s.unsolved / s.n:.2%})")
        parts += ["```", ""]
    return "\n".join(parts)


def write_report(
    out_dir: Path,
    stem: str,
    title: str,
    summaries: Sequence[Summary],
    thresholds: Sequence[int] = (),
    show_cost: bool = False,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = out_dir / f"{stem}.md", out_dir / f"{stem}.json"
    md_path.write_text(markdown_report(title, summaries, thresholds, show_cost))
    json_path.write_text(json.dumps([s.to_dict() for s in summaries], indent=2) + "\n")
    return md_path, json_path
