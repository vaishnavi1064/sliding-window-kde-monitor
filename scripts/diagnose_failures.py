"""Is the failure signature present in the features we feed the sketch?

Before investing in a better detector it is worth checking whether the signal
exists in the inputs at all. This compares every channel's behaviour in the
24 hours before each documented failure against normal operation, and reports a
standardized effect size per channel per failure.

Two specific hypotheses it tests:

1. The seven **analog** channels are what the sketch currently sees. If their
   effect sizes are small, no density-based detector on them can work well.
2. The eight **digital** channels are currently excluded, on the grounds that
   binary flags would dominate the geometry. But an air leak should make the
   compressor run *more*, and compressor duty cycle is a digital channel
   (`COMP`) aggregated over time. Excluding it may have discarded the signal.

Usage:  python -m scripts.diagnose_failures
"""

import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

from evaluation.labels import DEFAULT_HORIZON
from streaming.config import SETTINGS
from streaming.failures import FAILURES
from streaming.features import ANALOG_COLUMNS, DIGITAL_COLUMNS


def effect_sizes(
    frame: pd.DataFrame, columns: tuple[str, ...], horizon: timedelta
) -> pd.DataFrame:
    """Standardized mean difference per channel per failure, vs normal operation."""
    times = frame["timestamp"]

    in_any = np.zeros(len(frame), dtype=bool)
    for event in FAILURES:
        in_any |= ((times >= event.start - horizon) & (times <= event.end)).to_numpy()
    normal = frame.loc[~in_any, list(columns)]

    rows = []
    for event in FAILURES:
        mask = (times >= event.start - horizon) & (times < event.start)
        pre = frame.loc[mask.to_numpy(), list(columns)]
        if pre.empty:
            continue
        record = {"failure": event.name, "readings": len(pre)}
        for column in columns:
            spread = normal[column].std()
            if spread < 1e-9:
                record[column] = 0.0
            else:
                record[column] = (pre[column].mean() - normal[column].mean()) / spread
        rows.append(record)
    return pd.DataFrame(rows)


def duty_cycles(frame: pd.DataFrame, horizon: timedelta) -> pd.DataFrame:
    """Fraction of time each digital signal is asserted, pre-failure vs normal.

    Duty cycle is the aggregate a per-reading binary flag cannot express -- and
    is where a leak should show up: a leaking unit runs its compressor more.
    """
    times = frame["timestamp"]
    in_any = np.zeros(len(frame), dtype=bool)
    for event in FAILURES:
        in_any |= ((times >= event.start - horizon) & (times <= event.end)).to_numpy()

    normal = frame.loc[~in_any, list(DIGITAL_COLUMNS)].mean()
    rows = [{"period": "normal", **{c: normal[c] for c in DIGITAL_COLUMNS}}]
    for event in FAILURES:
        mask = ((times >= event.start - horizon) & (times < event.start)).to_numpy()
        pre = frame.loc[mask, list(DIGITAL_COLUMNS)]
        if pre.empty:
            continue
        rows.append({"period": f"pre-{event.name}", **{c: pre[c].mean() for c in DIGITAL_COLUMNS}})
    return pd.DataFrame(rows).set_index("period")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    args = parser.parse_args()
    horizon = timedelta(hours=args.horizon_hours)

    frame = pd.read_parquet(SETTINGS.parquet_path)
    print(f"{len(frame):,} readings, horizon {args.horizon_hours:.0f}h\n")

    print("=== analog channels: standardized mean shift in the pre-failure window ===")
    print("(units are standard deviations of normal operation; |d| > 0.8 is a large effect)\n")
    analog = effect_sizes(frame, ANALOG_COLUMNS, horizon)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(analog.round(2).to_string(index=False))
    numeric = analog[list(ANALOG_COLUMNS)]
    print("\nlargest |effect| per failure:")
    for _, row in analog.iterrows():
        magnitudes = {c: abs(row[c]) for c in ANALOG_COLUMNS}
        best = sorted(magnitudes.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {row['failure']}: " + ", ".join(f"{c} {row[c]:+.2f}" for c, _ in best))
    print(f"\nmean |effect| across all analog channels and failures: "
          f"{numeric.abs().to_numpy().mean():.2f}")
    print(f"max  |effect| anywhere:                                 "
          f"{numeric.abs().to_numpy().max():.2f}")

    print("\n\n=== digital channels: duty cycle (fraction of readings asserted) ===")
    print("(currently EXCLUDED from the sketch's feature vector)\n")
    duty = duty_cycles(frame, horizon)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(duty.round(3).to_string())

    baseline = duty.loc["normal"]
    print("\nrelative change vs normal, per failure:")
    for period in duty.index:
        if period == "normal":
            continue
        changes = {
            c: (duty.loc[period, c] - baseline[c]) / baseline[c]
            for c in DIGITAL_COLUMNS
            if baseline[c] > 1e-6
        }
        top = sorted(changes.items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  {period}: " + ", ".join(f"{c} {v * 100:+.0f}%" for c, v in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
