"""Compare detectors at equal alarm budgets.

Two problems make a naive comparison meaningless, and this script exists to
avoid both.

**Fixed thresholds are not comparable across methods.** SW-AKDE and RACE produce
scores on different scales (RACE's un-windowed counts grow without bound), so
"threshold 1.3" means something different to each. Operating points are
therefore chosen as *quantiles of each method's own score distribution*, which
compares them at the same alerting fraction -- the same alarm budget an operator
would actually be willing to absorb.

**A high alarm rate detects everything for free.** With a 24-hour horizon around
each failure, a detector alarming once a day lands inside a horizon most of the
time by construction. Every operating point is therefore calibrated against a
random detector emitting the same number of alarms.

Usage:
    python -m scripts.compare_methods
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import chance_detection, chance_p_value, evaluate
from scripts.run_evaluation import cache_path, rescore

# Alarm budgets, as the fraction of time the detector is alerting.
ALERTING_FRACTIONS = (0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)


def load(method: str) -> pd.DataFrame | None:
    path: Path = cache_path(method)
    return pd.read_parquet(path) if path.exists() else None


def compare(methods: list[str]) -> None:
    for method in methods:
        series = load(method)
        if series is None:
            print(f"\n=== {method}: no cached series, skipping ===")
            continue

        scores = rescore(series)
        timestamps = series["timestamp"]
        active = scores[scores > 0]

        print(f"\n=== {method} ===")
        print(f"  {len(scores):,} samples, {len(active):,} after burn-in; "
              f"score median={np.median(active):.2f} max={active.max():.2f}")
        print(f"\n{'alerting':>9} {'thresh':>9} {'events':>7} {'alarms/day':>11} "
              f"{'mean lead':>10} {'chance':>7} {'p':>7}  leads (h)")
        print("-" * 96)

        for fraction in ALERTING_FRACTIONS:
            threshold = float(np.quantile(scores, 1.0 - fraction))
            result = evaluate(timestamps, scores, threshold)
            total_alarms = result.false_alarms + result.detected_count
            counts = chance_detection(timestamps, total_alarms)
            expected = float(np.mean(counts)) if np.asarray(counts).size else 0.0
            p = chance_p_value(counts, result.detected_count)
            leads = " ".join(
                f"{e.lead_hours:+.0f}" if e.detected else "miss" for e in result.events
            )
            mean_lead = (
                f"{result.mean_lead_hours:+.1f}h"
                if result.mean_lead_hours is not None
                else "-"
            )
            print(f"{fraction * 100:>8.1f}% {threshold:>9.2f} "
                  f"{result.detected_count:>5}/{len(result.events)} "
                  f"{result.false_alarms_per_day:>11.2f} {mean_lead:>10} "
                  f"{expected:>7.2f} {p:>7.3f}  {leads}")

    print("\nHow to read this:")
    print("  Methods are compared at equal alerting fractions, not equal thresholds,")
    print("  because their score scales differ. 'chance' is how many of the four")
    print("  failures a random detector with the same alarm count would catch, and")
    print("  'p' is the fraction of random trials matching or beating the real one.")
    print("  A low p at a small alerting fraction is the only combination that")
    print("  demonstrates genuine detection rather than a generous alarm budget.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["swakde", "race"])
    args = parser.parse_args()
    compare(args.methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
