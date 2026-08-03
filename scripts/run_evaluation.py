"""Phase 3: evaluate the detector against MetroPT-3's four documented failures.

Structure matters here for practical reasons. Replaying 1.5M readings through
the sketch takes ~10 minutes, so the expensive pass runs **once per method** and
its score series is cached to Parquet; every threshold sweep afterwards is
instant. That keeps the slow step out of the analysis loop.

Methods:
  swakde  our sliding-window sketch (the thing under test)
  race    plain un-windowed RACE -- the baseline the paper itself compares to,
          and the one that isolates what the sliding window actually buys
  exact   brute-force exact KDE over the last N readings. The quantity the
          sketch approximates, computed without any sketch at all, so the gap
          between it and swakde is exactly what bounded memory costs in
          detection quality. Uses O(N x d) memory and O(N) work per query,
          which is precisely what the sketch exists to avoid -- it is a
          reference point, not a deployable alternative.

TAKDE was considered as a third baseline and dropped: the reference
implementation is a one-dimensional synthetic benchmark script (1-D histogram
binning, ground truth assumed to be a known Gaussian, error computed inline),
so running it on 7-D sensor data would require inventing a multivariate
window-selection statistic it does not provide. The result would measure our
variant rather than the published method. See docs/EVALUATION.md.

Usage:
    python -m scripts.run_evaluation --method swakde
    python -m scripts.run_evaluation --method race
    python -m scripts.run_evaluation --method swakde --report-only
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import chance_detection, chance_p_value, format_result, sweep
from sketch.p_stable import l2_collision_probability_vectorized
from sketch.race import RACE
from sketch.sw_akde import SlidingWindowEuclideanKDE
from streaming.config import SETTINGS
from streaming.features import ANALOG_COLUMNS, WarmupStandardizer
from streaming.scoring import RollingAnomalyScorer

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def cache_path(method: str) -> Path:
    return CACHE_DIR / (
        f"eval_{method}_rows{SETTINGS.rows}_k{SETTINGS.k}_w{SETTINGS.window_size}.parquet"
    )


class ExactWindowedKDE:
    """Brute-force sliding-window KDE: the quantity the sketch approximates.

    Keeps the last `window_size` standardized readings in a ring buffer and, at
    query time, sums the p-stable collision probability against every one of
    them. Same kernel and bandwidth as the Euclidean sketch, so the two are
    directly comparable -- the only difference is that this stores the data.
    """

    def __init__(self, window_size: int, dim: int, width: float, k: int):
        self.window_size = window_size
        self.width = width
        self.k = k
        self.buffer = np.zeros((window_size, dim))
        self.count = 0
        self.cursor = 0

    def update(self, x, t: int | None = None) -> None:
        self.buffer[self.cursor] = x
        self.cursor = (self.cursor + 1) % self.window_size
        self.count = min(self.count + 1, self.window_size)

    def query(self, x, t: int | None = None) -> float:
        if self.count == 0:
            return 0.0
        window = self.buffer[: self.count]
        distances = np.sqrt(np.maximum(((window - x) ** 2).sum(axis=1), 0.0))
        probabilities = l2_collision_probability_vectorized(distances, self.width)
        # ** k for the k concatenated hashes, matching the sketch's cell code
        # and compute_true_kde_l2 (Finding H).
        return float((probabilities**self.k).sum())


def build_score_series(method: str, frame: pd.DataFrame) -> pd.DataFrame:
    raw = frame[list(ANALOG_COLUMNS)].to_numpy(dtype=float)
    timestamps = frame["timestamp"].to_numpy()

    rng = np.random.default_rng(0)
    if method == "swakde":
        model = SlidingWindowEuclideanKDE(
            rows=SETTINGS.rows,
            k=SETTINGS.k,
            dim=len(ANALOG_COLUMNS),
            width=SETTINGS.lsh_width,
            window_size=SETTINGS.window_size,
            eh_relative_error=SETTINGS.eh_relative_error,
            rng=rng,
        )
        window_size = SETTINGS.window_size
    elif method == "exact":
        model = ExactWindowedKDE(
            window_size=SETTINGS.window_size,
            dim=len(ANALOG_COLUMNS),
            width=SETTINGS.lsh_width,
            k=SETTINGS.k,
        )
        window_size = SETTINGS.window_size
    elif method == "race":
        # Un-windowed RACE: same LSH geometry, integer counters, nothing ever
        # expires. Isolates the contribution of sliding-window semantics.
        model = RACE(rows=SETTINGS.rows, k=SETTINGS.k, dim=len(ANALOG_COLUMNS), rng=rng)
        window_size = SETTINGS.window_size
    else:
        raise ValueError(f"unknown method: {method}")

    standardizer = WarmupStandardizer(len(ANALOG_COLUMNS), warmup=SETTINGS.warmup)
    scorer = RollingAnomalyScorer()

    out_times: list = []
    out_density: list[float] = []
    out_score: list[float] = []

    clock = 0
    started = time.perf_counter()

    for index in range(len(raw)):
        values = raw[index]
        standardizer.observe(values)
        if not standardizer.fitted:
            continue

        x = standardizer.transform(values)
        if not np.all(np.isfinite(x)):
            continue

        clock += 1
        if method == "race":
            model.update(x)
        else:
            model.update(x, clock)

        if clock % SETTINGS.query_every:
            continue

        density = model.query(x) if method == "race" else model.query(x, clock)
        # Same gate as the live consumer: density is not comparable across time
        # until the window is full (for RACE, until an equivalent amount of
        # history has accumulated, so both methods get the same burn-in).
        score = scorer.score(density) if clock >= window_size else 0.0

        out_times.append(timestamps[index])
        out_density.append(density)
        out_score.append(score)

        if len(out_times) % 20_000 == 0:
            rate = clock / (time.perf_counter() - started)
            remaining = (len(raw) - index) / max(rate, 1) / 60
            print(f"  {index:,}/{len(raw):,} readings "
                  f"({rate:,.0f}/s, ~{remaining:.0f} min left)", flush=True)

    elapsed = time.perf_counter() - started
    print(f"  done in {elapsed / 60:.1f} min ({clock / elapsed:,.0f} readings/s)")

    return pd.DataFrame(
        {"timestamp": out_times, "density": out_density, "score": out_score}
    )


def report(series: pd.DataFrame, method: str) -> None:
    scores = series["score"].to_numpy()
    valid = scores > 0
    print(f"\n=== {method} ===")
    print(f"score samples: {len(scores):,} ({int(valid.sum()):,} after burn-in)")
    if valid.any():
        active = scores[valid]
        print(f"  score  median={np.median(active):.2f}  p95={np.percentile(active, 95):.2f}  "
              f"p99={np.percentile(active, 99):.2f}  max={active.max():.2f}")
    density = series["density"].to_numpy()
    print(f"  density median={np.median(density):.1f}  "
          f"range={density.min():.1f}-{density.max():.1f}")

    thresholds = np.array([0.5, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    print(f"\n{'thresh':>7} {'events':>7} {'alarms/day':>11} {'alarms':>7} "
          f"{'alerting':>9} {'mean lead':>10} {'chance':>7} {'p':>6}  leads (h)")
    print("-" * 104)
    for result in sweep(series["timestamp"], scores, thresholds):
        leads = " ".join(
            f"{e.lead_hours:+.0f}" if e.detected else "miss" for e in result.events
        )
        mean_lead = (
            f"{result.mean_lead_hours:+.1f}h" if result.mean_lead_hours is not None else "-"
        )
        # Calibrate against luck: the same number of alarms placed at random.
        counts = chance_detection(
            series["timestamp"], result.false_alarms + result.detected_count
        )
        expected = float(np.mean(counts)) if not isinstance(counts, float) else 0.0
        p = chance_p_value(counts, result.detected_count)
        print(f"{result.threshold:>7.2f} {result.detected_count:>5}/4 "
              f"{result.false_alarms_per_day:>11.2f} {result.false_alarms:>7} "
              f"{result.alerting_fraction * 100:>8.1f}% {mean_lead:>10} "
              f"{expected:>7.2f} {p:>6.3f}  {leads}")

    print("\n  'chance' = events a random detector with the same alarm count would catch;")
    print("  'p' = fraction of random trials matching or beating this detector.")
    print("  A detector that alarms often catches a 24h horizon easily -- p is what")
    print("  separates real detection from a high alarm rate.")

    print("\nDetail at the operating threshold chosen in Phase 2 (1.3):")
    print(format_result(sweep(series["timestamp"], scores, np.array([1.3]))[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("swakde", "race", "exact"), default="swakde")
    parser.add_argument("--records", type=int, default=0, help="0 = whole dataset")
    parser.add_argument("--report-only", action="store_true",
                        help="use the cached score series, do not replay")
    args = parser.parse_args()

    path = cache_path(args.method)
    if args.report_only or path.exists():
        if not path.exists():
            raise SystemExit(f"no cache at {path}; run without --report-only first")
        print(f"Using cached score series {path.name}")
        series = pd.read_parquet(path)
    else:
        if not SETTINGS.parquet_path.exists():
            raise SystemExit("run scripts.download_data first")
        frame = pd.read_parquet(SETTINGS.parquet_path)
        if args.records:
            frame = frame.iloc[: args.records]
        print(f"Replaying {len(frame):,} readings through {args.method} "
              f"(rows={SETTINGS.rows}, k={SETTINGS.k}, window={SETTINGS.window_size}) ...")
        series = build_score_series(args.method, frame)
        series.to_parquet(path, index=False)
        print(f"Cached score series -> {path.name}")

    report(series, args.method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
