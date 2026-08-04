"""Native core vs Python core: throughput, latency, memory (Phase 5).

Run with `make bench-native`.

Four things this is careful about, because the whole point of the phase is a
speedup claim:

1. **The baseline is the optimized Python core, not a naive one.** Phase 1
   already bought 2.9x by vectorizing the hashing (docs/PERFORMANCE.md); quoting
   that again here would be double-counting.
2. **Throughput and latency are measured in separate passes.** Timing every
   individual update to build a latency distribution costs a `perf_counter` pair
   per update, which is a few percent of a native update. Instrumenting the
   throughput run would understate it, so the throughput pass is uninstrumented.
3. **Each measurement runs in its own process.** Resident-set size is only
   meaningful as a delta, and a Python run's garbage would otherwise still be
   sitting in the heap while the native run is measured. `memory_bytes()` is
   self-reported by each core and so cannot be the only evidence for a memory
   claim; RSS from a clean process is the independent check.
4. **Memory is measured on the real stream at the real pipeline settings.**
   The Euclidean kernel with `hash_range = 2^20` is what streaming/config.py
   runs and what produces cell counts in the hundreds of thousands; the angular
   kernel at k=3 can only ever occupy `rows * 8` cells, which would make the
   comparison look tidy and mean nothing. This mirrors scripts/memory_check.py
   so the figures line up with the Phase 3 table.

The hashing stays in Python in both cores by design -- it is one matmul and fell
out of the profile after Phase 1 -- so the report also measures what share of
native update time it now accounts for. That share is the ceiling on any further
speedup from this approach, and stating it keeps the result honest.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np

from sketch.native import AVAILABLE, compiler
from sketch.sw_akde import SlidingWindowAngularKDE, SlidingWindowEuclideanKDE
from streaming.config import SETTINGS
from streaming.features import ANALOG_COLUMNS, WarmupStandardizer

# MetroPT-3: 15 sensor channels, 1,516,948 readings sampled every 10s (0.1 Hz).
METROPT_DIM = 15
METROPT_RECORDS = 1_516_948

ROW_SWEEP = (100, 200, 400, 800, 1600, 3200)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rss_bytes() -> int | None:
    """Resident set size of this process, without adding a dependency.

    psutil would be one import for one number; CLAUDE.md section 10 asks for a
    specific reason per dependency and this does not clear that bar.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # argtypes are not optional here. GetCurrentProcess returns the
        # pseudo-handle (HANDLE)-1, which ctypes marshals as a 32-bit int
        # without them; the call then fails, returning 0 and setting no error
        # code -- which reads exactly like "this platform cannot report RSS".
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.WorkingSetSize if ok else None

    try:  # Linux
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return None  # macOS and anything else: reported as n/a


def measure_throughput(backend: str, rows: int, k: int, dim: int, window: int, n: int) -> dict:
    """One (backend, rows) point on the synthetic throughput workload."""
    rng = np.random.default_rng(0)
    stream = rng.normal(size=(n, dim))
    queries = rng.normal(size=(100, dim))

    sketch = SlidingWindowAngularKDE(
        rows=rows,
        k=k,
        dim=dim,
        window_size=window,
        rng=np.random.default_rng(1),
        backend=backend,
    )

    # Pass 1: throughput, uninstrumented.
    start = time.perf_counter()
    for t, x in enumerate(stream, start=1):
        sketch.update(x, t)
    update_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for q in queries:
        sketch.query(q, t=n)
    query_seconds = time.perf_counter() - start

    # Pass 2: per-update latency distribution, on an already-warm sketch.
    latencies = np.empty(min(n, 2000))
    clock = n
    for i in range(len(latencies)):
        clock += 1
        started = time.perf_counter()
        sketch.update(stream[i % n], clock)
        latencies[i] = (time.perf_counter() - started) * 1e6

    # Pass 3: what the deliberately-still-in-Python hashing costs, so the
    # remaining headroom is visible rather than implied.
    start = time.perf_counter()
    for x in stream:
        sketch.hashes.codes(x)
    hash_seconds = time.perf_counter() - start

    return {
        "backend": sketch.backend,
        "rows": rows,
        "updates_per_second": n / update_seconds,
        "queries_per_second": len(queries) / query_seconds,
        "latency_p50_us": float(np.percentile(latencies, 50)),
        "latency_p99_us": float(np.percentile(latencies, 99)),
        "hash_share": hash_seconds / update_seconds,
    }


def _pipeline_stream(records: int) -> tuple[np.ndarray, str]:
    """Standardized readings for the memory run: real MetroPT-3 if downloaded.

    Falls back to a drifting synthetic stream, which keeps the script runnable
    without the dataset but produces a different cell population -- so the
    report says which one was used rather than presenting them alike.
    """
    dim = len(ANALOG_COLUMNS)
    if SETTINGS.parquet_path.exists():
        import pandas as pd

        frame = pd.read_parquet(SETTINGS.parquet_path, columns=list(ANALOG_COLUMNS))
        raw = frame.to_numpy(dtype=float)[: records + SETTINGS.warmup]
        del frame

        standardizer = WarmupStandardizer(dim, warmup=SETTINGS.warmup)
        readings = []
        for values in raw:
            standardizer.observe(values)
            if not standardizer.fitted:
                continue
            readings.append(standardizer.transform(values))
            if len(readings) >= records:
                break
        return np.asarray(readings), "MetroPT-3"

    rng = np.random.default_rng(0)
    drift = np.arange(records).reshape(-1, 1) * 0.001
    return rng.normal(size=(records, dim)) + drift, "synthetic (MetroPT-3 not downloaded)"


def measure_memory(backend: str, records: int) -> dict:
    """Cell-array memory and throughput at the streaming pipeline's settings.

    Same kernel, rows, window and standardizer as streaming/config.py, so this is
    directly comparable to the cell-reclamation table in docs/PERFORMANCE.md.
    """
    stream, source = _pipeline_stream(records)

    # Build the stream before snapshotting RSS, and drop pandas' buffers, so the
    # delta covers the sketch rather than the data loading.
    gc.collect()
    before = rss_bytes()

    sketch = SlidingWindowEuclideanKDE(
        rows=SETTINGS.rows,
        k=SETTINGS.k,
        dim=stream.shape[1],
        width=SETTINGS.lsh_width,
        window_size=SETTINGS.window_size,
        eh_relative_error=SETTINGS.eh_relative_error,
        rng=np.random.default_rng(0),
        backend=backend,
    )

    started = time.perf_counter()
    for t, x in enumerate(stream, start=1):
        sketch.update(x, t)
    elapsed = time.perf_counter() - started

    after = rss_bytes()
    return {
        "backend": sketch.backend,
        "source": source,
        "readings": len(stream),
        "updates_per_second": len(stream) / elapsed,
        "cells": sketch.cell_count,
        "cell_bytes": sketch.memory_bytes(),
        "reclaimed": sketch.cells_reclaimed,
        "rss_delta_bytes": (after - before) if (before is not None and after is not None) else None,
    }


def measure_tail(rows: int, k: int, dim: int, window: int) -> dict:
    """Is the native p99 tail the once-per-window compaction sweep?

    The arithmetic invites the conclusion -- compaction runs 1 update in
    `window`, so at window=256 that is 0.4% of updates, landing exactly inside
    the top 1%. Backs the claim in docs/PERFORMANCE.md that it nonetheless is
    not the cause, by measuring the same distribution with compaction
    effectively disabled.
    """
    rng = np.random.default_rng(0)
    stream = rng.normal(size=(6000, dim))

    out = {}
    for label, compact_every in (("with_compaction", None), ("without_compaction", 10**9)):
        sketch = SlidingWindowAngularKDE(
            rows=rows, k=k, dim=dim, window_size=window,
            rng=np.random.default_rng(1), backend="native", compact_every=compact_every,
        )
        for t, x in enumerate(stream[:2000], start=1):  # warm up
            sketch.update(x, t)

        latencies = np.empty(4000)
        clock = 2000
        for i in range(len(latencies)):
            clock += 1
            started = time.perf_counter()
            sketch.update(stream[i % len(stream)], clock)
            latencies[i] = (time.perf_counter() - started) * 1e6

        out[label] = {
            "p50_us": float(np.percentile(latencies, 50)),
            "p99_us": float(np.percentile(latencies, 99)),
            "p999_us": float(np.percentile(latencies, 99.9)),
            "max_us": float(latencies.max()),
            "compactions": sketch.compactions,
        }
    return out


def measure_eval_parity(records: int) -> dict:
    """Both cores through the whole Phase 3 evaluation pipeline, on real data.

    The strongest form of the accuracy claim: not just that the sketches agree,
    but that the standardizer -> duty cycle -> sketch -> rolling scorer chain
    produces the same `density` and `score` series, compared with
    `pandas.Series.equals` rather than a tolerance. Calls build_score_series
    directly, so it never touches the cached evaluation parquet files.
    """
    import pandas as pd

    from scripts.run_evaluation import build_score_series

    if not SETTINGS.parquet_path.exists():
        raise SystemExit("run `python -m scripts.download_data` first")

    frame = pd.read_parquet(SETTINGS.parquet_path).iloc[:records]

    series, rates = {}, {}
    for backend in ("python", "native"):
        started = time.perf_counter()
        series[backend] = build_score_series("swakde", frame, "analog", backend)
        rates[backend] = len(frame) / (time.perf_counter() - started)

    python, cpp = series["python"], series["native"]
    mismatched = [
        column for column in python.columns if not python[column].equals(cpp[column])
    ]
    return {
        "readings": len(frame),
        "scored_rows": len(python),
        "columns": list(python.columns),
        "mismatched_columns": mismatched,
        "identical": not mismatched,
        "python_readings_per_second": rates["python"],
        "native_readings_per_second": rates["native"],
    }


def run_child(mode: str, backend: str, **arguments) -> dict:
    """Measure one point in a clean process, so RSS deltas do not contaminate."""
    command = [sys.executable, "-m", "scripts.benchmark_native", "--single",
               "--mode", mode, "--backend", backend]
    for name, value in arguments.items():
        command += [f"--{name.replace('_', '-')}", str(value)]

    completed = subprocess.run(
        command, capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _megabytes(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1e6:,.1f}"


def report_throughput(k: int, dim: int, window: int, n: int) -> list[dict]:
    print("Throughput and latency (synthetic stream, angular kernel)")
    print(
        f"{'rows':>6} {'py upd/s':>10} {'native upd/s':>13} {'speedup':>8} "
        f"{'py qry/s':>10} {'native qry/s':>13} {'speedup':>8} "
        f"{'p50 us':>8} {'p99 us':>8} {'hash%':>7}"
    )
    print("-" * 104)

    results = []
    for rows in ROW_SWEEP:
        shared = dict(rows=rows, k=k, dim=dim, window=window, n=n)
        python = run_child("throughput", "python", **shared)
        cpp = run_child("throughput", "native", **shared)
        results.append({"python": python, "native": cpp})
        print(
            f"{rows:>6} {python['updates_per_second']:>10,.0f} "
            f"{cpp['updates_per_second']:>13,.0f} "
            f"{cpp['updates_per_second'] / python['updates_per_second']:>7.1f}x "
            f"{python['queries_per_second']:>10,.0f} {cpp['queries_per_second']:>13,.0f} "
            f"{cpp['queries_per_second'] / python['queries_per_second']:>7.1f}x "
            f"{cpp['latency_p50_us']:>8.1f} {cpp['latency_p99_us']:>8.1f} "
            f"{cpp['hash_share'] * 100:>6.1f}%"
        )

    print(
        "\np50/p99 are native per-update latency. 'hash%' is the share of native update"
        "\ntime spent in the Python hash bank, which both cores share -- the ceiling on"
        "\nany further speedup from porting the cell array alone."
    )
    return results


def report_memory(records: int) -> tuple[dict, bool]:
    print(
        f"\n\nMemory at pipeline settings (Euclidean kernel, rows={SETTINGS.rows}, "
        f"k={SETTINGS.k}, window={SETTINGS.window_size}, {records:,} readings)"
    )
    measurements = {}
    for backend in ("python", "native"):
        measurements[backend] = run_child("memory", backend, records=records)

    print(f"data: {measurements['native']['source']}\n")
    print(
        f"{'core':>8} {'cells':>10} {'cell MB':>10} {'RSS delta MB':>14} "
        f"{'bytes/cell':>11} {'upd/s':>9} {'full pass':>11}"
    )
    print("-" * 80)
    for backend, point in measurements.items():
        # Time to stream all of MetroPT-3 once at these settings.
        full_pass_minutes = METROPT_RECORDS / point["updates_per_second"] / 60
        print(
            f"{backend:>8} {point['cells']:>10,} {_megabytes(point['cell_bytes']):>10} "
            f"{_megabytes(point['rss_delta_bytes']):>14} "
            f"{point['cell_bytes'] / max(point['cells'], 1):>11,.0f} "
            f"{point['updates_per_second']:>9,.0f} {full_pass_minutes:>9.1f}m"
        )

    python, cpp = measurements["python"], measurements["native"]
    print(f"\ncell memory: {python['cell_bytes'] / max(cpp['cell_bytes'], 1):.1f}x smaller")
    if python["rss_delta_bytes"] and cpp["rss_delta_bytes"]:
        print(f"RSS delta:   {python['rss_delta_bytes'] / cpp['rss_delta_bytes']:.1f}x smaller")
    print(
        "'cell MB' is each core's own accounting of its cell array (a sys.getsizeof walk"
        "\nvs. sizeof plus inferred per-node overhead), so the two are estimates of one"
        "\nquantity by different means; the RSS delta is the independent check."
    )

    consistent = python["cells"] == cpp["cells"]
    if not consistent:
        print(
            f"\nWARNING: cell counts differ ({python['cells']:,} vs {cpp['cells']:,}); they "
            "must be identical. Investigate before quoting any of this.",
            file=sys.stderr,
        )
    return measurements, consistent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("throughput", "memory"), default="throughput")
    parser.add_argument(
        "--tail-check",
        action="store_true",
        help="test whether the native p99 tail is the compaction sweep, and exit",
    )
    parser.add_argument(
        "--eval-parity",
        action="store_true",
        help="run both cores through the Phase 3 evaluation pipeline on real data, and exit",
    )
    parser.add_argument("--backend", default="python")
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dim", type=int, default=METROPT_DIM)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument(
        "--records",
        type=int,
        default=150_000,
        help="readings for the memory comparison at pipeline settings",
    )
    parser.add_argument("--skip-memory", action="store_true")
    arguments = parser.parse_args()

    if arguments.single:
        if arguments.mode == "memory":
            point = measure_memory(arguments.backend, arguments.records)
        else:
            point = measure_throughput(
                arguments.backend, arguments.rows, arguments.k,
                arguments.dim, arguments.window, arguments.n,
            )
        print(json.dumps(point))
        return 0

    if not AVAILABLE:
        print("native core is not built -- run `make native` first", file=sys.stderr)
        return 1

    if arguments.tail_check:
        print(f"Native per-update latency at rows={arguments.rows}, window={arguments.window}\n")
        tail = measure_tail(arguments.rows, arguments.k, arguments.dim, arguments.window)
        print(f"{'':>20} {'p50':>9} {'p99':>9} {'p99.9':>9} {'max':>9} {'compactions':>12}")
        for label, point in tail.items():
            print(
                f"{label:>20} {point['p50_us']:>8.1f}u {point['p99_us']:>8.1f}u "
                f"{point['p999_us']:>8.1f}u {point['max_us']:>8.1f}u "
                f"{point['compactions']:>12,}"
            )
        with_it, without = tail["with_compaction"], tail["without_compaction"]
        change = abs(with_it["p99_us"] - without["p99_us"]) / without["p99_us"] * 100
        print(f"\np99 changes by {change:.0f}% when compaction is removed.")
        return 0

    if arguments.eval_parity:
        print(f"Phase 3 evaluation pipeline, both cores, {arguments.records:,} real readings\n")
        result = measure_eval_parity(arguments.records)
        print(f"scored rows: {result['scored_rows']:,}   columns: {result['columns']}")
        print(f"python: {result['python_readings_per_second']:,.0f} readings/s")
        print(f"native: {result['native_readings_per_second']:,.0f} readings/s")
        speedup = result["native_readings_per_second"] / result["python_readings_per_second"]
        print(f"speedup: {speedup:.1f}x")
        if result["identical"]:
            print("\nidentical: every column matches exactly (pandas.Series.equals)")
            return 0
        print(f"\nMISMATCH in columns: {result['mismatched_columns']}", file=sys.stderr)
        return 1

    print(f"native core: {compiler()}   python: {sys.version.split()[0]}")
    print(f"dim={arguments.dim} (MetroPT-3 channels), k={arguments.k}, "
          f"window={arguments.window}, n={arguments.n:,}\n")

    report_throughput(arguments.k, arguments.dim, arguments.window, arguments.n)
    if arguments.skip_memory:
        return 0
    _, consistent = report_memory(arguments.records)
    return 0 if consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
