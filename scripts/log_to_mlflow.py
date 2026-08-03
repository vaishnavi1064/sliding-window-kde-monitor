"""Log the evaluation to MLflow.

One run per (method, alerting fraction), so operating points are comparable
across methods in the UI rather than only inside a printed table. Figures and
their CSVs are attached to a parent run.

Tracking goes to a local SQLite database, `mlflow.db` (gitignored). MLflow 3.x
puts the filesystem store in maintenance mode and refuses it by default, so a
database backend is the supported local option. Inspect with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Usage:  python -m scripts.log_to_mlflow
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from evaluation.metrics import chance_detection, chance_p_value, evaluate
from scripts.run_evaluation import cache_path, rescore
from streaming.config import SETTINGS

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES = REPO_ROOT / "docs" / "figures"
ALERTING_FRACTIONS = (0.20, 0.10, 0.05, 0.02, 0.01)
EXPERIMENT = "swakde-metropt-evaluation"


def sketch_params(method: str, feature_set: str) -> dict:
    return {
        "method": method,
        "feature_set": feature_set,
        "rows": SETTINGS.rows,
        "k": SETTINGS.k,
        "window_size": SETTINGS.window_size,
        "eh_relative_error": SETTINGS.eh_relative_error,
        "lsh_width": SETTINGS.lsh_width,
        "query_every": SETTINGS.query_every,
        "warmup": SETTINGS.warmup,
        "horizon_hours": 24,
    }


def log_method(method: str, feature_set: str) -> int:
    path = cache_path(method, feature_set)
    if not path.exists():
        print(f"  {method} [{feature_set}]: no cached series, skipping")
        return 0

    series = pd.read_parquet(path)
    scores = rescore(series)
    timestamps = series["timestamp"]
    logged = 0

    for fraction in ALERTING_FRACTIONS:
        threshold = float(np.quantile(scores, 1.0 - fraction))
        result = evaluate(timestamps, scores, threshold)
        total = result.false_alarms + result.detected_count
        counts = chance_detection(timestamps, total)
        p = chance_p_value(counts, result.detected_count)

        with mlflow.start_run(run_name=f"{method}-{feature_set}-{fraction:.3f}"):
            mlflow.log_params({**sketch_params(method, feature_set),
                               "alerting_fraction": fraction,
                               "threshold": round(threshold, 4)})
            mlflow.log_metrics({
                "events_detected": result.detected_count,
                "events_total": len(result.events),
                "false_alarms": result.false_alarms,
                "false_alarms_per_day": result.false_alarms_per_day,
                "normal_days": result.normal_days,
                "alerting_fraction_actual": result.alerting_fraction,
                "p_value_vs_chance": p,
                "chance_expected_detections": (
                    float(np.mean(counts)) if np.asarray(counts).size else 0.0
                ),
            })
            if result.mean_lead_hours is not None:
                mlflow.log_metric("mean_lead_hours", result.mean_lead_hours)
            # Per-event detail: four events is few enough to log individually,
            # and per-event is how this has to be read anyway.
            for event in result.events:
                mlflow.log_metric(f"detected_{event.name}", int(event.detected))
                if event.lead_hours is not None:
                    mlflow.log_metric(f"lead_hours_{event.name}", event.lead_hours)
            logged += 1

    print(f"  {method} [{feature_set}]: {logged} operating points")
    return logged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["swakde", "exact", "race"])
    parser.add_argument("--features", default="analog")
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT)
    print(f"Logging to experiment '{EXPERIMENT}' ...")

    total = 0
    for method in args.methods:
        total += log_method(method, args.features)

    # Figures and their CSVs on a parent run, so the artifacts live with the
    # numbers rather than only on disk.
    if FIGURES.exists():
        with mlflow.start_run(run_name="figures"):
            mlflow.log_params(sketch_params("all", args.features))
            for item in sorted(FIGURES.iterdir()):
                if item.suffix in {".png", ".csv"}:
                    mlflow.log_artifact(str(item), artifact_path="figures")
        print(f"  figures: {len(list(FIGURES.iterdir()))} artifacts")

    print(f"\n{total} operating points logged. Browse with:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
