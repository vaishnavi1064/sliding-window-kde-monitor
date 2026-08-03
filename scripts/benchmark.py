"""Throughput benchmark for the SW-AKDE sketch.

Establishes the Python baseline the C++ core (Phase 5) will be measured against.
For that comparison to be honest the baseline has to be a reasonably optimized
Python implementation, not a naive one -- see AngularHashBank.
"""

import time

import numpy as np

from sketch.sw_akde import SlidingWindowAngularKDE

# MetroPT-3: 15 sensor channels, 1,516,948 readings sampled every 10s (0.1 Hz).
METROPT_DIM = 15
METROPT_ROWS = 1_516_948


def benchmark(rows: int, k: int, dim: int, window_size: int, n: int) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    stream = rng.normal(size=(n, dim))
    queries = rng.normal(size=(100, dim))
    sketch = SlidingWindowAngularKDE(
        rows=rows, k=k, dim=dim, window_size=window_size, rng=np.random.default_rng(1)
    )

    start = time.perf_counter()
    for t, x in enumerate(stream, start=1):
        sketch.update(x, t)
    update_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for q in queries:
        sketch.query(q, t=n)
    query_seconds = time.perf_counter() - start

    return n / update_seconds, len(queries) / query_seconds


def main() -> None:
    k, window_size, n = 5, 256, 3000

    print(f"dim={METROPT_DIM} (MetroPT-3 channels), k={k}, window={window_size}, n={n}\n")
    print(f"{'rows':>6}  {'updates/s':>12}  {'queries/s':>12}  {'us/update':>10}  {'full pass':>10}")
    print("-" * 60)

    for rows in (100, 200, 400, 800, 1600, 3200):
        updates_per_second, queries_per_second = benchmark(
            rows=rows, k=k, dim=METROPT_DIM, window_size=window_size, n=n
        )
        micros = 1e6 / updates_per_second
        full_pass_minutes = METROPT_ROWS / updates_per_second / 60
        print(
            f"{rows:>6}  {updates_per_second:>12,.0f}  {queries_per_second:>12,.0f}  "
            f"{micros:>10.0f}  {full_pass_minutes:>8.1f}m"
        )

    print("\n'full pass' = time to stream all of MetroPT-3 once at that sketch size.")


if __name__ == "__main__":
    main()
