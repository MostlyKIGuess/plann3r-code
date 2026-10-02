"""Score the online stopping rule over the episodes of one run_nav.py run.

Each episode is a true positive when the agent stopped within the success
distance, a false positive when it stopped farther away, a false negative when
it came that close but never stopped, and a true negative otherwise. Writes
stopping_predictions.csv and stopping_metrics.json (accuracy, precision, recall,
non-oracle success rate, SPL, SSPL) into the run folder. baseline/evaluate.sh
runs it after the no-oracle-paper mode.

Usage:
    pixi run python stopping_condition/summarize_online_stopping.py RUN_DIR --success-distance=1.0
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("--success-distance", type=float, default=1.0)
    args = parser.parse_args()

    rows = []
    for episode_dir in sorted(path for path in args.base_dir.iterdir() if path.is_dir()):
        results_path = episode_dir / "results.csv"
        if not results_path.exists():
            continue

        results = pd.read_csv(results_path)
        distances = results["distance_to_goal"].dropna()
        min_distance = float(distances.min()) if len(distances) else float("inf")
        stop_path = episode_dir / "online_stopping.json"
        stopped = stop_path.exists()
        stop_data = json.loads(stop_path.read_text()) if stopped else {}
        stopped_distance = stop_data.get("distance_to_goal")

        if stopped and stopped_distance < args.success_distance:
            outcome = "TP"
        elif stopped:
            outcome = "FP"
        elif min_distance < args.success_distance:
            outcome = "FN"
        else:
            outcome = "TN"

        rows.append(
            {
                "episode": episode_dir.name,
                "outcome": outcome,
                "stopped": stopped,
                "stopped_distance": stopped_distance,
                "min_distance": min_distance,
                **stop_data,
            }
        )

    predictions = pd.DataFrame(rows)
    counts = {key: int((predictions["outcome"] == key).sum()) for key in ("TP", "TN", "FP", "FN")}
    total = len(predictions)
    metrics = {
        **counts,
        "total": total,
        "accuracy": (counts["TP"] + counts["TN"]) / total if total else 0.0,
        "precision": counts["TP"] / (counts["TP"] + counts["FP"])
        if counts["TP"] + counts["FP"]
        else 0.0,
        "recall": counts["TP"] / (counts["TP"] + counts["FN"])
        if counts["TP"] + counts["FN"]
        else 0.0,
        "non_oracle_success_rate": counts["TP"] / total if total else 0.0,
    }

    navigation_summary_path = args.base_dir / "results_summary.csv"
    if navigation_summary_path.exists():
        navigation_summary = pd.read_csv(navigation_summary_path)
        metrics["spl"] = float(navigation_summary["spl"].mean())
        metrics["sspl"] = float(navigation_summary["sspl"].mean())

    predictions_path = args.base_dir / "stopping_predictions.csv"
    metrics_path = args.base_dir / "stopping_metrics.json"
    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("\n" + "=" * 50)
    print("LIVE STOPPING METRICS")
    print("=" * 50)
    print(f"Total Runs evaluated: {total}")
    for key in ("TP", "TN", "FP", "FN"):
        print(f"{key}: {counts[key]}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Non-oracle navigation success rate: {metrics['non_oracle_success_rate']:.4f}")
    if "spl" in metrics:
        print(f"SPL:  {metrics['spl']:.4f} ({100.0 * metrics['spl']:.2f}%)")
        print(f"SSPL: {metrics['sspl']:.4f} ({100.0 * metrics['sspl']:.2f}%)")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
