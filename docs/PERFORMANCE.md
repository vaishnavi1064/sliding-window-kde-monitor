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
