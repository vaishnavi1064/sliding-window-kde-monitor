"""Offline replay of MetroPT-3 through the sketch, for choosing detector parameters.

Running the full Docker stack takes minutes per configuration, which is far too
slow to choose a parameter honestly -- and choosing one by intuition is how you
end up tuning until a demo passes. This replays the real data through the real
sketch in-process, so a parameter sweep takes seconds and the choice can be made
from measured separation between normal operation and a fault.

It is also the seed of the Phase 3 evaluation harness: the same loop, pointed at
the four documented failures instead of an injected one, is what produces
detection lead times.

Usage:
    python -m scripts.tune_detector
    python -m scripts.tune_detector --records 40000 --inject-at 20000
"""

import argparse

import numpy as np
import pandas as pd

from sketch.sw_akde import SlidingWindowEuclideanKDE
from streaming.config import SETTINGS
from streaming.features import ANALOG_COLUMNS, WarmupStandardizer
from streaming.scoring import RollingAnomalyScorer


def density_series(
    frame: pd.DataFrame,
    inject_at: int,
    inject_duration: int,
    inject_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the frame, returning (density, injected_flag) per query."""
    raw = frame[list(ANALOG_COLUMNS)].to_numpy(dtype=float)

    sketch = SlidingWindowEuclideanKDE(
        rows=SETTINGS.rows,
        k=SETTINGS.k,
        dim=len(ANALOG_COLUMNS),
        width=SETTINGS.lsh_width,
        window_size=SETTINGS.window_size,
        eh_relative_error=SETTINGS.eh_relative_error,
        rng=np.random.default_rng(0),
    )
    standardizer = WarmupStandardizer(len(ANALOG_COLUMNS), warmup=SETTINGS.warmup)

    densities: list[float] = []
    injected: list[bool] = []
    clock = 0

    for index, values in enumerate(raw):
        is_injected = inject_at <= index < inject_at + inject_duration
        if is_injected:
            values = values * inject_scale

        standardizer.observe(values)
        if not standardizer.fitted:
            continue

        x = standardizer.transform(values)
        clock += 1
        sketch.update(x, clock)

        if clock % SETTINGS.query_every == 0 and clock >= sketch.window_size:
            densities.append(sketch.query(x, clock))
            injected.append(is_injected)

    return np.array(densities), np.array(injected)


def evaluate(
    densities: np.ndarray,
    injected: np.ndarray,
    smoothing: int,
    min_history: int,
    log_space: bool = False,
) -> dict:
    scorer = RollingAnomalyScorer(
        history=2000, min_history=min_history, smoothing=smoothing
    )
    values = np.log1p(densities) if log_space else densities
    scores = np.array([scorer.score(v) for v in values])

    # Only score once the baseline is established, or the warm-up zeros
    # flatter the false-alarm figure.
    valid = np.arange(len(scores)) >= min_history
    normal = scores[valid & ~injected]
    fault = scores[valid & injected]

    peak_fault = float(fault.max()) if fault.size else 0.0
    max_normal = float(normal.max()) if normal.size else 0.0

    # Choose the threshold from the data rather than assuming 3: the geometric
    # midpoint between the worst normal excursion and the fault peak leaves
    # proportional headroom on both sides. Its usefulness depends on how long
    # the score stays above it -- a tall but brief spike cannot satisfy an
    # alert's `for` duration.
    threshold = float(np.sqrt(max(peak_fault, 1e-9) * max(max_normal, 1e-9)))
    sustained = int((fault > threshold).sum())

    return {
        "smoothing": smoothing,
        "peak_fault": peak_fault,
        "max_normal": max_normal,
        "threshold": threshold,
        "separation": peak_fault / max(max_normal, 1e-9),
        "sustained_samples": sustained,
        # query_every readings per density sample, replayed at 60 readings/s.
        "sustained_seconds": sustained * SETTINGS.query_every / 60.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=30000)
    parser.add_argument("--inject-at", type=int, default=16000)
    parser.add_argument("--inject-duration", type=int, default=12000)
    parser.add_argument("--inject-scale", type=float, default=4.0)
    args = parser.parse_args()

    if not SETTINGS.parquet_path.exists():
        raise SystemExit(f"{SETTINGS.parquet_path} not found; run scripts.download_data")

    frame = pd.read_parquet(SETTINGS.parquet_path).iloc[: args.records]
    print(f"Replaying {len(frame):,} readings "
          f"(fault injected at {args.inject_at:,} for {args.inject_duration:,}) ...")

    densities, injected = density_series(
        frame, args.inject_at, args.inject_duration, args.inject_scale
    )
    print(f"  {len(densities):,} density samples, {int(injected.sum()):,} during the fault\n")

    normal_d, fault_d = densities[~injected], densities[injected]
    for name, values in (("normal", normal_d), ("fault", fault_d)):
        if values.size:
            print(
                f"  density {name:6}: median={np.median(values):7.2f}  "
                f"iqr={np.percentile(values,25):6.2f}-{np.percentile(values,75):6.2f}  "
                f"range={values.min():6.2f}-{values.max():7.2f}"
            )

    best = None
    for log_space in (False, True):
        label = "log density" if log_space else "raw density"
        print(f"\n=== {label} ===")
        print(f"{'smoothing':>10}  {'peak(fault)':>12}  {'max(normal)':>12}  "
              f"{'separation':>11}  {'threshold':>10}  {'sustained':>12}")
        print("-" * 78)
        for smoothing in (10, 20, 40, 80, 160, 320):
            min_history = max(200, 5 * smoothing)
            if len(densities) < min_history * 2:
                continue
            result = evaluate(densities, injected, smoothing, min_history, log_space)
            result["log_space"] = log_space
            print(f"{result['smoothing']:>10}  {result['peak_fault']:>12.2f}  "
                  f"{result['max_normal']:>12.2f}  {result['separation']:>11.1f}x  "
                  f"{result['threshold']:>10.2f}  "
                  f"{result['sustained_seconds']:>9.0f}s")
            # A usable configuration needs both clear separation AND a signal
            # that persists longer than the alert's `for` duration.
            usable = result["separation"] >= 2.0 and result["sustained_seconds"] >= 40
            key = (usable, result["separation"])
            best_usable = (
                best is not None
                and best["separation"] >= 2.0
                and best["sustained_seconds"] >= 40
            )
            if best is None or key > (best_usable, best["separation"]):
                best = result

    if best:
        space = "log" if best["log_space"] else "raw"
        print(f"\nbest: {space} density, smoothing={best['smoothing']}")
        print(f"  fault peaks at {best['peak_fault']:.2f}, normal never exceeds "
              f"{best['max_normal']:.2f} ({best['separation']:.1f}x separation)")
        print(f"  threshold {best['threshold']:.2f} is cleared for "
              f"{best['sustained_seconds']:.0f}s of wall clock at 60 readings/s")
        if best["sustained_seconds"] < 40:
            print("  WARNING: signal is too brief for a 30s `for` duration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
