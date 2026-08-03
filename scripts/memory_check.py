"""Measure what cell reclamation buys, on real data.

Quantifies the memory half of Finding F: without compaction the sparse cell
dictionary grows one entry per distinct cell ever visited, because a cold cell
is never touched again and so never prunes itself.

Usage:  python -m scripts.memory_check --records 400000
"""

import argparse
import gc
import sys
import time

import numpy as np
import pandas as pd

from sketch.sw_akde import SlidingWindowEuclideanKDE
from streaming.config import SETTINGS
from streaming.features import ANALOG_COLUMNS, WarmupStandardizer


def deep_size(sketch: SlidingWindowEuclideanKDE) -> int:
    """Approximate bytes held by the cell dictionary and its histograms."""
    total = sys.getsizeof(sketch.cells)
    for key, eh in sketch.cells.items():
        total += sys.getsizeof(key) + sys.getsizeof(eh)
        total += sys.getsizeof(eh.buckets)
        total += sum(sys.getsizeof(b) for b in eh.buckets)
    return total


def run(frame: pd.DataFrame, compact: bool) -> dict:
    raw = frame[list(ANALOG_COLUMNS)].to_numpy(dtype=float)
    sketch = SlidingWindowEuclideanKDE(
        rows=SETTINGS.rows,
        k=SETTINGS.k,
        dim=len(ANALOG_COLUMNS),
        width=SETTINGS.lsh_width,
        window_size=SETTINGS.window_size,
        eh_relative_error=SETTINGS.eh_relative_error,
        rng=np.random.default_rng(0),
        compact_every=None if compact else 10**12,
    )
    standardizer = WarmupStandardizer(len(ANALOG_COLUMNS), warmup=SETTINGS.warmup)

    gc.collect()
    clock = 0
    started = time.perf_counter()
    early_rate = None

    for values in raw:
        standardizer.observe(values)
        if not standardizer.fitted:
            continue
        clock += 1
        sketch.update(standardizer.transform(values), clock)
        if clock == 50_000:
            early_rate = clock / (time.perf_counter() - started)

    elapsed = time.perf_counter() - started
    return {
        "compact": compact,
        "readings": clock,
        "seconds": elapsed,
        "rate": clock / elapsed,
        "early_rate": early_rate or clock / elapsed,
        "cells": len(sketch.cells),
        "bytes": deep_size(sketch),
        "reclaimed": sketch.cells_reclaimed,
    }


def sketch_bytes_by_rows(frame: pd.DataFrame, row_counts: tuple[int, ...]) -> None:
    """Does the sketch actually save memory here? Compare against storing the window.

    The sketch exists so you never have to hold the window. Exact windowed KDE
    holds `window_size x dim` floats -- which is *tiny* when the window is short
    and the data low-dimensional. The sketch's own footprint, by contrast, is
    independent of dimension but scales with rows. So there is a crossover
    dimension below which storing the raw window is simply cheaper, and it is
    worth knowing which side of it a given application sits on.
    """
    raw = frame[list(ANALOG_COLUMNS)].to_numpy(dtype=float)
    dim = len(ANALOG_COLUMNS)
    exact_bytes = SETTINGS.window_size * dim * 8

    print(f"\n\nsketch footprint vs storing the window "
          f"(window={SETTINGS.window_size}, dim={dim})")
    print(f"exact windowed KDE holds {SETTINGS.window_size} x {dim} floats = "
          f"{exact_bytes / 1e6:.2f} MB\n")
    print(f"{'rows':>6} {'cells':>10} {'sketch':>11} {'vs exact':>10} "
          f"{'crossover dim':>14}")
    print("-" * 58)

    for rows in row_counts:
        sketch = SlidingWindowEuclideanKDE(
            rows=rows,
            k=SETTINGS.k,
            dim=dim,
            width=SETTINGS.lsh_width,
            window_size=SETTINGS.window_size,
            eh_relative_error=SETTINGS.eh_relative_error,
            rng=np.random.default_rng(0),
        )
        standardizer = WarmupStandardizer(dim, warmup=SETTINGS.warmup)
        clock = 0
        for values in raw:
            standardizer.observe(values)
            if not standardizer.fitted:
                continue
            clock += 1
            sketch.update(standardizer.transform(values), clock)

        size = deep_size(sketch)
        # Dimension at which storing the window would cost as much as the sketch.
        crossover = size / (SETTINGS.window_size * 8)
        print(f"{rows:>6} {len(sketch.cells):>10,} {size / 1e6:>8.1f} MB "
              f"{size / exact_bytes:>9.0f}x {crossover:>13,.0f}")

    print("\nThe sketch's footprint does not depend on dimension; exact storage grows")
    print("with it. Below the crossover dimension, holding the raw window is cheaper")
    print("than sketching it.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=400_000)
    parser.add_argument("--rows-sweep", action="store_true",
                        help="compare sketch footprint against storing the window")
    args = parser.parse_args()

    frame = pd.read_parquet(SETTINGS.parquet_path).iloc[: args.records]
    print(f"{len(frame):,} readings, rows={SETTINGS.rows}, window={SETTINGS.window_size}\n")

    if args.rows_sweep:
        sketch_bytes_by_rows(frame, (50, 100, 200, 400))
        return 0

    results = []
    for compact in (False, True):
        label = "with compaction" if compact else "without compaction"
        print(f"running {label} ...", flush=True)
        results.append(run(frame, compact))

    print(f"\n{'':<20} {'cells':>12} {'cell memory':>13} {'rate':>12} "
          f"{'early rate':>12} {'slowdown':>9}")
    print("-" * 84)
    for r in results:
        label = "with compaction" if r["compact"] else "without compaction"
        slowdown = r["early_rate"] / r["rate"]
        print(f"{label:<20} {r['cells']:>12,} {r['bytes'] / 1e6:>10.0f} MB "
              f"{r['rate']:>9,.0f}/s {r['early_rate']:>9,.0f}/s {slowdown:>8.2f}x")

    without, with_ = results
    print(f"\ncells      {without['cells'] / max(with_['cells'], 1):>6.1f}x fewer with compaction")
    print(f"cell memory{without['bytes'] / max(with_['bytes'], 1):>6.1f}x smaller")
    print(f"throughput {with_['rate'] / max(without['rate'], 1):>6.1f}x faster")
    print(f"reclaimed  {with_['reclaimed']:,} dead cells over the run")
    print("\n'early rate' is throughput over the first 50k readings; the gap between")
    print("it and the overall rate is the deceleration caused by unbounded growth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
