#!/usr/bin/env python3
"""Summarize the planner ablations against the paper checkpoint.

Per-episode success, SPL and SSPL are read from the `results_summary.txt` that
run_nav.py writes, so the summary uses the same numbers as the run logs. Every
ablation is also scored on the episodes it shares with `paper` for that task,
so a run that dropped episodes cannot look better by averaging over an easier
subset. Writes ablation_summary.csv and ablation_summary.md under --root.
baseline/evaluate.sh runs it at the end unless SUMMARIZE=false.

Usage:
    pixi run python baseline/summarize_ablation.py --root "$RESULTS_ROOT"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

REFERENCE = "paper"
TASKS = ("imitate", "reverse", "altgoal", "shortcut")
# Result folders under RESULTS_ROOT that are not ground-truth-localization ablations.
NOT_ABLATIONS = {"episode_lists", "megaloc", "no-oracle-paper"}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def episode_key(directory: Path) -> str:
    suffix = "_vggt_nav_topological_pixelwise"
    name = directory.name
    return name[: -len(suffix)] if name.endswith(suffix) else name


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg if isinstance(cfg, dict) else {}


def run_is_valid(run_dir: Path, task: str) -> bool:
    """Alt-goal runs count only when scored against the annotated object."""
    if task != "altgoal":
        return True
    cfg = load_yaml(run_dir / "config.yaml")
    if not cfg:
        raise FileNotFoundError(f"Alt-goal run has no config.yaml: {run_dir}")
    return cfg.get("task_type") == "alt_goal_v2" and cfg.get("goal_position_method") == "semantic_instance"


def episode_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def collect(root: Path, task: str) -> dict[str, dict[str, float]]:
    """Collect per-episode metrics from one ablation's task directory.

    When an episode was run more than once, the newest summary is used.
    """
    latest: dict[str, Path] = {}
    valid_runs: dict[Path, bool] = {}
    for summary_path in root.rglob("results_summary.txt"):
        episode_dir = summary_path.parent
        run_dir = episode_dir.parent
        if run_dir not in valid_runs:
            valid_runs[run_dir] = run_is_valid(run_dir, task)
        if not valid_runs[run_dir]:
            continue
        key = episode_key(episode_dir)
        if key not in latest or summary_path.stat().st_mtime_ns > latest[key].stat().st_mtime_ns:
            latest[key] = summary_path

    rows: dict[str, dict[str, float]] = {}
    for key, summary_path in latest.items():
        values = episode_summary(summary_path)
        rows[key] = {
            "success": float(values["success_status"] == "success"),
            "spl": float(values["spl"]),
            "sspl": float(values["sspl"]),
        }
    return rows


def discover_ablations(root: Path) -> list[str]:
    names = sorted(entry.name for entry in root.iterdir() if entry.is_dir() and entry.name not in NOT_ABLATIONS)
    # Keep the reference first so the markdown reads top-down.
    if REFERENCE in names:
        names.remove(REFERENCE)
        names.insert(0, REFERENCE)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    ablations = discover_ablations(args.root)
    reference = (
        {task: collect(args.root / REFERENCE / task, task) for task in TASKS}
        if REFERENCE in ablations
        else {task: {} for task in TASKS}
    )

    rows = []
    markdown = [
        "# Planner ablations (ground-truth localization)",
        "",
        "All rows use ground-truth topological localization, the paper episode "
        "lists, controller and 300-step budget. Only the planner checkpoint and "
        "its propagation maps change. `vs paper` columns are computed on the "
        "episodes an ablation shares with the paper checkpoint. A blank means the "
        "paper run is absent from this results root.",
        "",
        "| Ablation | Task | Episodes | SR | SPL | SSPL | Paired | SR vs paper | "
        "SPL vs paper | SSPL vs paper |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for ablation in ablations:
        for task in TASKS:
            task_dir = args.root / ablation / task
            if not task_dir.is_dir():
                continue
            runs = collect(task_dir, task)
            if not runs:
                continue
            own = {metric: mean([runs[key][metric] for key in runs]) for metric in ("success", "spl", "sspl")}

            base = reference.get(task, {})
            paired = sorted(set(runs) & set(base))
            deltas: dict[str, float] = {}
            if paired and ablation != REFERENCE:
                for metric in ("success", "spl", "sspl"):
                    deltas[metric] = mean([runs[key][metric] for key in paired]) - mean(
                        [base[key][metric] for key in paired]
                    )

            rows.append(
                {
                    "ablation": ablation,
                    "task": task,
                    "episodes": len(runs),
                    "success_rate": 100 * own["success"],
                    "spl": 100 * own["spl"],
                    "sspl": 100 * own["sspl"],
                    "paired_with_paper": len(paired) if ablation != REFERENCE else "-",
                    "success_rate_vs_paper": 100 * deltas.get("success", 0.0) if deltas else "",
                    "spl_vs_paper": 100 * deltas.get("spl", 0.0) if deltas else "",
                    "sspl_vs_paper": 100 * deltas.get("sspl", 0.0) if deltas else "",
                }
            )

            def cell(metric: str) -> str:
                return f"{100 * deltas[metric]:+.2f}" if deltas else "-"

            markdown.append(
                f"| {ablation} | {task} | {len(runs)} | {100*own['success']:.2f} | "
                f"{100*own['spl']:.2f} | {100*own['sspl']:.2f} | "
                f"{len(paired) if ablation != REFERENCE else '-'} | "
                f"{cell('success')} | {cell('spl')} | {cell('sspl')} |"
            )

    # Task-averaged view. Each ablation is averaged over the tasks it ran.
    markdown += [
        "",
        "Task averages (unweighted over the tasks each ablation completed):",
        "",
        "| Ablation | Tasks | Avg SR | Avg SPL | Avg SSPL |",
        "|---|---:|---:|---:|---:|",
    ]
    for ablation in ablations:
        own_rows = [row for row in rows if row["ablation"] == ablation]
        if not own_rows:
            continue
        markdown.append(
            f"| {ablation} | {len(own_rows)} | "
            f"{mean([r['success_rate'] for r in own_rows]):.2f} | "
            f"{mean([r['spl'] for r in own_rows]):.2f} | "
            f"{mean([r['sspl'] for r in own_rows]):.2f} |"
        )

    csv_path = args.root / "ablation_summary.csv"
    md_path = args.root / "ablation_summary.md"
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "ablation", "task", "episodes", "success_rate", "spl", "sspl",
            "paired_with_paper", "success_rate_vs_paper", "spl_vs_paper", "sspl_vs_paper",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text("\n".join(markdown) + "\n")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
