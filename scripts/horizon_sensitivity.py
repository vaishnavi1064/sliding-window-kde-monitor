"""How much does the prediction horizon drive the result?

The horizon is a parameter of the *evaluation*, not of the detector, and it cuts
both ways. A longer horizon gives a detector more credit for early warning, but
it also makes accidental detection far easier: a random detector alarming once a
day lands inside a 24-hour window most of the time, and inside a 3-hour window
rarely. So the same detector can look convincing or worthless depending purely on
this choice, which is exactly why it has to be reported rather than fixed
silently.

Reads cached score series, so this is instant.

Usage:  python -m scripts.horizon_sensitivity
"""

import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

from evaluation.metrics import chance_detection, chance_p_value, evaluate
from scripts.run_evaluation import cache_path, rescore

HORIZONS_HOURS = (3, 6, 12, 24, 48)
ALERTING_FRACTIONS = (0.10, 0.05, 0.02, 0.01, 0.005)


def analyse(method: str, feature_set: str) -> None:
    path = cache_path(method, feature_set)
    if not path.exists():
        print(f"\n=== {method} [{feature_set}]: no cached series, skipping ===")
        return

    series = pd.read_parquet(path)
    scores = rescore(series)
    timestamps = series["timestamp"]

    print(f"\n=== {method} [{feature_set}] ===")
    print(f"{'horizon':>8} {'alerting':>9} {'thresh':>8} {'events':>7} "
          f"{'alarms/day':>11} {'mean lead':>10} {'chance':>7} {'p':>7}  leads (h)")
    print("-" * 100)

    best = None
    for hours in HORIZONS_HOURS:
        horizon = timedelta(hours=hours)
        for fraction in ALERTING_FRACTIONS:
            threshold = float(np.quantile(scores, 1.0 - fraction))
            result = evaluate(timestamps, scores, threshold, horizon=horizon)
            total = result.false_alarms + result.detected_count
            counts = chance_detection(timestamps, total, horizon=horizon)
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
            print(f"{hours:>7}h {fraction * 100:>8.1f}% {threshold:>8.2f} "
                  f"{result.detected_count:>5}/{len(result.events)} "
                  f"{result.false_alarms_per_day:>11.2f} {mean_lead:>10} "
                  f"{expected:>7.2f} {p:>7.3f}  {leads}")
            if best is None or (p, -result.detected_count) < (best[0], -best[1]):
                best = (p, result.detected_count, hours, fraction, mean_lead)
        print()

    if best:
        p, detected, hours, fraction, mean_lead = best
        print(f"  strongest evidence: {detected}/4 at a {hours}h horizon, "
              f"alerting {fraction * 100:.1f}% of the time, p={p:.3f} (lead {mean_lead})")
        if p >= 0.05:
            print("  -- still not significant at the conventional 0.05 level")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["exact", "race", "swakde"])
    parser.add_argument("--features", default="analog")
    args = parser.parse_args()
    for method in args.methods:
        analyse(method, args.features)

    print("\nNote: scanning horizons and thresholds and then quoting the best p is")
    print("multiple comparisons over four events. Treat these as exploratory --")
    print("the honest headline is the pre-registered operating point, not the best cell.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
