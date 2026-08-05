# Performance baseline

Engineering log for Lever 1 (CLAUDE.md §2). Everything here is measured, not estimated.
Reproduce with `make bench` (throughput) or the profiling snippets described below.

Workload unless stated otherwise: `dim=15` (MetroPT-3 channel count), `k=5`, `window=256`,
synthetic Gaussian stream. Machine: Windows 11, CPython 3.12.10, numpy on one core.

## Step 1 — profile before optimizing

`cProfile` over 2000 updates at `rows=400`, original scalar implementation:

```
15,180,865 function calls in 6.646 seconds

  ncalls  tottime  function
 4000000    2.882  angular_hash.py:eval
  800000    1.297  exponential_histogram.py:add
  800000    0.785  sw_akde.py:_cell_code
    2000    0.423  sw_akde.py:update
  800000    0.331  exponential_histogram.py:_expire
 4000000    0.257  numpy multiarray.dot
  812800    0.066  math.ceil          <-- recomputing a constant, 800k times
```

Split: **hashing ~59%**, exponential histogram ~31%.

The cost was not arithmetic. Each update issued `rows * k` = 2000 separate `np.dot` calls on
15-element vectors; per-call overhead dominated the 15 multiply-adds inside. `math.ceil` also
appeared 812,800 times computing `ceil(k/2)+1`, which never changes.

## Step 2 — two changes

1. **`AngularHashBank`** (`sketch/angular_hash.py`) — all `rows * k` hyperplanes live in one
   `(rows*k, dim)` matrix, so one matmul yields every bit, and a second small matmul packs each
   row's `k` bits into a cell code. The scalar `AngularHash` is kept as the readable reference
   and pinned to the bank by `tests/test_hash_bank.py`.
2. **Hoisted the merge cap** into `ExponentialHistogram.__init__`.

## Step 3 — profile after

Same workload, same machine:

```
5,599,665 function calls in 2.319 seconds

  ncalls  tottime  function
  800000    1.064  exponential_histogram.py:add
    2000    0.362  sw_akde.py:update
  800000    0.313  exponential_histogram.py:_expire
  800000    0.178  dict.get
  376632    0.097  exponential_histogram.py:_evict_oldest
  800000    0.080  buckets.py:__init__
```

**6.646s → 2.319s, a 2.9x speedup**, with all validation tiers still green (the test suite
itself went 14.4s → 3.5s).

Hashing has dropped out of the profile entirely. The exponential histogram is now **~72%** of
update time.

## Throughput (`make bench`)

| rows | updates/s | queries/s | µs/update | one full MetroPT-3 pass |
|---:|---:|---:|---:|---:|
| 100 | 9,909 | 23,315 | 101 | 2.6 min |
| 200 | 4,681 | 12,862 | 214 | 5.4 min |
| 400 | 2,005 | 5,276 | 499 | 12.6 min |
| 800 | 795 | 2,297 | 1,258 | 31.8 min |
| 1600 | 286 | 788 | 3,493 | 88.3 min |
| 3200 | 112 | 320 | 8,964 | 226.6 min |

## Cell reclamation — the memory half of Finding F

Found by running the full dataset rather than a benchmark: throughput decayed
from ~810 to ~300 updates/s part way through 1.5M readings while the process
reached **2.7 GB**. Not a leak in the ordinary sense — a consequence of the
sketch's own design.

Expiry is lazy and per-cell: a histogram only prunes itself when something
touches it. A cell that goes cold is never touched again, so it keeps its
buckets forever. The published space bound `O(RW/eps · log²N)` counts a **dense**
`R × W` array, where an unused cell costs nothing extra. Every real
implementation — the reference's and ours — stores cells **sparsely**, because
almost all of them are empty. Nothing then bounds the dictionary: it gains one
entry per distinct cell ever visited, so memory tracks *readings seen* rather
than *window size*.

`SlidingWindowKDE.compact` drops fully expired cells once per window. That is
semantically free (such a cell contributes exactly zero to any query) and O(1)
amortised per update. Measured over 400,000 MetroPT-3 readings at `rows=400`,
`window=3600` (`make memcheck`):

| | cells | cell memory | throughput | deceleration |
|---|---:|---:|---:|---:|
| without compaction | 1,859,317 | 666 MB | 794/s | 1.34x |
| with compaction | 156,644 | 83 MB | 922/s | 1.18x |
| | **11.9x fewer** | **8.0x smaller** | **1.2x faster** | |

4.8M dead cells were reclaimed over the run. The important number is not the
ratio but the *shape*: with compaction the live cell count settles at a level
set by the window, while without it the count grows with every reading ever
seen — which is what took the full-dataset run to 2.7 GB. Without this, the
sublinear-memory claim does not survive contact with a sparse implementation.

## What this means downstream

- **Phase 2 (streaming) is not throughput-constrained.** MetroPT-3 arrives every 10s (0.1 Hz); even at
  `rows=3200` we are ~112x faster than real time, and ~2000x at `rows=400`. Accelerated replay
  is what makes a full pass slow, not the sketch.
- **Phase 3 (evaluation) is throughput-constrained.** A single pass at `rows=3200` is nearly
  4 hours, and the paper's sweeps multiply that across row and window settings. Plan sweeps
  around this or run them against the native core.
- **Phase 5 (C++) should target `ExponentialHistogram`, not the hashing.** That is now settled
  by measurement rather than assumption: the EH update path is ~72% of the time, and it is the
  scalar, branchy, allocation-heavy part (a `Bucket` object per element, list mutation, the
  cascading merge loop) that vectorization cannot reach. The hashing is already a matmul and
  would gain little from being rewritten.
- The C++ benchmark must be quoted against *this* baseline, not the pre-optimization one.
  Claiming the 2.9x already banked here as part of a native-core speedup would be dishonest.

---

# Phase 5 — the native core

C++17 behind pybind11, in `src/`, selected with `backend="native"`. Same machine as above; native
core built by MSVC 19.44 at `/O2`.

Every number in this section comes from a committed script, and each subsection names the command
that regenerates it:

```bash
make native                                              # compile the core
make bench-native                                        # throughput, latency, memory tables
python -m scripts.benchmark_native --tail-check           # the p99 tail attribution
python -m scripts.benchmark_native --eval-parity --records 30000   # end-to-end parity on real data
```

(`make native` and `make bench-native` are `python -m scripts.build_native` and
`python -m scripts.benchmark_native`; the numbers below were produced by those module invocations,
as `make` is not installed on this machine.)

**What was ported, and what deliberately was not.** The prediction logged above was followed:
the cell array moved to C++ and the hashing stayed in NumPy. Concretely, C++ now owns the per-row
loop, the cell lookup and every exponential histogram; Python still computes the cell codes with
the two matmuls of `AngularHashBank`. One pybind11 crossing per update, with the GIL released for
the row loop.

Two representational changes carry most of the win, and neither is a semantic change:

1. **A `Bucket` is 16 bytes of POD, not a 56-byte `PyObject` plus an 8-byte list slot.** There is
   one bucket per element inside a cell's window, so this is where both the memory and the
   allocator traffic were.
2. **Buckets live in a power-of-two ring buffer, not a list.** Eviction is a head-index bump
   instead of `list.pop(0)`, which is O(bucket count). In a dense stream almost every update
   evicts, so that shift was squarely on the hot path.

## Throughput and latency

Synthetic stream, angular kernel, `dim=15`, `k=5`, `window=256`, `n=3000`. The Python column is
re-measured in the same run, so each ratio is internally consistent rather than read off the
older table.

| rows | Python upd/s | native upd/s | speedup | Python qry/s | native qry/s | speedup | native p50 | native p99 | hash share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 9,741 | 100,054 | **10.3x** | 26,084 | 159,566 | 6.1x | 9.2 µs | 14.7 µs | 45.2% |
| 200 | 4,574 | 55,853 | **12.2x** | 13,061 | 105,831 | 8.1x | 16.1 µs | 27.2 µs | 35.3% |
| 400 | 2,009 | 27,887 | **13.9x** | 5,161 | 59,407 | 11.5x | 32.8 µs | 92.0 µs | 27.9% |
| 800 | 688 | 13,079 | **19.0x** | 2,058 | 28,643 | 13.9x | 69.9 µs | 130.6 µs | 22.0% |
| 1600 | 308 | 5,617 | **18.3x** | 914 | 10,550 | 11.5x | 159.2 µs | 346.3 µs | 17.4% |
| 3200 | 113 | 1,867 | **16.6x** | 371 | 4,246 | 11.4x | 457.6 µs | 923.6 µs | 15.1% |

**10–19x on updates, 6–14x on queries.** Quoted against the optimized Python core, so the 2.9x
banked in Phase 1 is *not* included — the two speedups multiply to ~30–55x against the original
scalar implementation, but that figure is not what this table reports.

**Run-to-run variance is real and worth stating.** An earlier run of the same script on the same
machine gave 10.9–22.1x on updates and 6.0–17.5x on queries: individual per-row ratios move by up
to ~3x between runs, because this is a desktop under a normal load rather than an isolated
benchmark host. The defensible claim is **~10–20x on updates and ~6–14x on queries**, not any
single cell of the table.

The *cell-memory* figures below are structural rather than timed, and did reproduce exactly across
both runs — 63.9 MB vs 29.0 MB, 761 vs 345 bytes per cell, to the byte. The RSS and throughput
columns in that table are measured and drift a little (RSS by under 1 MB, throughput as above).

**"hash share" is the honest ceiling.** It is the fraction of native update time still spent in
the Python hash bank that both cores share. At `rows=100` it is 45%: nearly half the remaining
time is code this phase deliberately did not touch, so no further work on the cell array can do
much there. It falls to 15% at `rows=3200`, which is broadly why the speedup grows with row count
— the larger the sketch, the more of the work is the part that was ported. Getting past ~20x means
porting the hashing too, and only then.

**On the p99 tail.** The obvious explanation is wrong, and was checked rather than assumed
(`--tail-check`). The compaction sweep runs once per window (1 update in 256, i.e. 0.4%), which
would land exactly inside the top 1% — but removing compaction barely moves the distribution: at
`rows=400`, p99 is 92.2 µs with compaction and 97.1 µs without, a 5% difference in the *opposite*
direction to the hypothesis. The ordering of the extreme values flips between runs (this run:
max 521 µs with, 269 µs without; an earlier run: 510 µs with, 618 µs without), which is itself the
point — the tail is ordinary scheduling and allocator jitter on a non-isolated desktop, not the
sweep. Compaction is genuinely O(1) amortised.

## Memory

Real MetroPT-3 readings through the Euclidean kernel at the settings the pipeline runs
(`streaming/config.py`: `rows=400`, `k=3`, `window=3600`), 150,000 readings. The angular kernel at
`k=3` can only ever occupy `rows * 8` cells, which would have made this table look tidy and mean
nothing.

| core | cells | cell memory | RSS delta | bytes/cell | upd/s | full pass |
|---|---:|---:|---:|---:|---:|---:|
| Python | 83,920 | 63.9 MB | 113.9 MB | 761 | 1,060 | 23.9 min |
| native | 83,920 | 29.0 MB | 54.8 MB | 345 | 22,659 | 1.1 min |
| | identical | **2.2x smaller** | **2.1x smaller** | | **21.4x** | |

Two things worth reading carefully:

- **The cell counts are identical**, which is the point. A memory win from holding fewer cells
  would be a semantic difference, not an engineering one.
- **`cell memory` and `RSS delta` are independent measurements that agree** (2.2x and 2.1x). Each
  core reports its own structural accounting — a `sys.getsizeof` walk on one side, `sizeof` plus
  inferred per-node overhead on the other — so neither is authoritative alone; RSS comes from the
  OS, per process, with each measurement in a freshly started interpreter. `bytes/cell` depends on
  how much data has been seen (more live cells share a roughly fixed bucket budget, so the figure
  falls as the run lengthens); the *ratio* is what transfers, not the absolute.

The native core does not shrink a histogram's ring buffer when its bucket count falls, so a cell
that was once busy keeps its capacity — a power of two, hence up to 2x its current need. Compaction
drops fully expired cells outright, which bounds the population; this is a note about what the
2.2x would be with shrinking, not a caveat on the measurement.

**21x on the real pipeline workload**, again against the optimized Python core — the number that
matters for Phase 3: a full 1,516,948-reading MetroPT-3 pass at these settings goes from
~24 minutes to ~66 seconds. As with the table above this ratio is timed and so moves between runs
(an earlier run gave 19.0x); treat it as ~20x.

## Accuracy

**Bit-for-bit identical, not "within tolerance".** Both cores accumulate integer bucket counts,
halve the oldest, and sum in row order, so every query is a sum of exactly representable doubles
added in the same sequence — there is no floating-point reordering to excuse a difference.
`tests/test_native_parity.py` asserts `==` on query output for both kernels, and compares state
cell by cell (bucket timestamps and sizes, `total`, `last`) plus the compaction schedule. A
tolerance-based check would have hidden exactly the bug class a reimplementation introduces: an
off-by-one in the merge cascade perturbs a count by one element and sails through 1%.

The native core is also checked against brute-force ground truth directly
(`test_native_core_matches_brute_force_truth`), so the Phase 5 accuracy gate does not rest solely
on a chain of reasoning through the Python core.

End to end, the claim holds where it actually matters. Running 30,000 real MetroPT-3 readings
through the whole Phase 3 evaluation pipeline — standardizer, duty cycle, sketch, rolling scorer —
the two backends produce **identical `timestamp`, `density` and `score` series**, compared with
`pandas.Series.equals` rather than a tolerance, at 15,356 readings/s versus 1,132 (13.6x). Swapping
the core does not move a single published number. Reproduce with
`python -m scripts.benchmark_native --eval-parity --records 30000`.

CI runs both: one job installs with `SWAKDE_SKIP_NATIVE=1` and asserts the native core is absent,
the other installs with `SWAKDE_REQUIRE_NATIVE=1` on Linux *and* Windows and asserts it is present.
Without the first, the pure-Python path would never be exercised — every GitHub Linux runner ships
g++. Without the second, a broken core would install quietly (the extension is declared `optional`)
and every parity test would skip while reporting green.

## Verification state

What has actually been observed, and what has not. Kept here rather than only in a commit message
so it cannot drift out of view.

| Claim | State | Evidence |
|---|---|---|
| Compiles under MSVC (local) | **Verified** | `MSVC 1944`, `/O2`, zero warnings at `/W3` |
| Compiles under gcc | **Verified** | CI run 30979739606, ubuntu-latest: `native core built: gcc 13.3.0` |
| Compiles under a second MSVC | **Verified** | same run, windows-latest: `native core built: MSVC 1951` |
| `pip install -e .` builds it (no custom build driver) | **Verified** | both native-core jobs: "Install with the native core" → success |
| `SWAKDE_SKIP_NATIVE=1` yields a working pure-Python install | **Verified** | python-core job: "Confirm the native core really is absent" → success |
| Parity suite green | **Verified locally only** | 130 pass with the extension, 93 pass + 37 skip without |
| **CI demonstrates parity** | **NOT verified** | all three jobs fail at `Test`; see below |
| **Native core loads in the container** | **NOT verified** | `docker compose build consumer` has never run |

**CI is red, for a reason that predates this work.** `pytest` aborts during collection on every
job:

```
tests/test_evaluation.py:4:  import pandas as pd  → ModuleNotFoundError: No module named 'pandas'
tests/test_mcp_server.py:12: import pandas as pd  → ModuleNotFoundError: No module named 'pandas'
Interrupted: 2 errors during collection
```

CI installs `.[dev]`, which is numpy plus pytest; those two modules import pandas at module scope.
This is not a Phase 5 regression — the repository has had three CI runs ever and all three failed
identically, including the two on `main` (`a5106cb`, `5dc5530`) that precede this branch. It is
left unfixed deliberately, so the failure stays visible rather than being folded into an unrelated
commit. The fix is one line (install `.[dev,streaming]`, or move pandas into the `dev` extra), but
it is a Phase 3/4 packaging defect and belongs in its own change.

The consequence for Phase 5 is specific and should not be overstated in either direction: CI
*proves the C++ compiles and installs on gcc and MSVC*, which is what it was added for, and does
*not yet prove the two cores agree*, because the tests never execute there. That comparison is
currently evidenced only by local runs.

## What is still true, and what changed

- **The Python core remains the default and the oracle** (CLAUDE.md §13). Results should not depend
  on whether the machine that produced them had a compiler, so `backend="python"` stays the library
  default; `"auto"` prefers native and falls back, `"native"` demands it and raises otherwise.
- **The streaming pipeline opts in** via `SKETCH_BACKEND` (default `auto`), and the consumer logs
  the resolved core at startup. The container installs g++ to build the extension and then removes
  it, because otherwise `auto` would silently mean "Python" in the one place the speed was wanted.
- **Phase 3's applicability finding stands, but the boundary did move.** `docs/EVALUATION.md`
  derives the crossover as linear in `bytes_per_cell`, so a 2.2x cheaper cell shifts it by exactly
  that factor: the rule of thumb goes from `dim > ~5 x rows` to `dim > ~2.3 x rows`, and the
  crossover dimension at `rows=400` from ~1,650 to ~750. MetroPT's **seven** channels remain two
  orders of magnitude below it, so the conclusion — exact windowed KDE is the right choice for this
  workload — is unchanged. But "unaffected" would have been the wrong word: a constant-factor win
  on a boundary that depends linearly on that constant does move it, just not nearly far enough to
  reach this application.
