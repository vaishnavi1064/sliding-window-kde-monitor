# Reference audit — line-level notes

Source: https://github.com/VedTheKnight/Streaming-ANN-and-Sliding-Window-KDE, branch `racecms-benchmark`, commit `cc7d7bc1f3902939071a7430a9187be8f05cb237` (2026-07-31). Archived at `C:\Users\chaug\reference-archives\Streaming-ANN-and-Sliding-Window-KDE-cc7d7bc.zip`. All paths below are relative to `Code/SlidingWindowKDE/` in that tree. Read `CLAUDE.md` first — this file is the line-level backup for its claims.

Verified 2026-08-02 by checking out the pinned commit and reading each file directly (not from memory of the paper or secondhand summaries).

## `buckets_DS.py` (13 lines) — reusable as-is

Trivial POD: `Bucket(t)` sets `size=1`, `tst=t`; getters/setters for both. Nothing to fix.

## `Exponential_Histogram.py` (59 lines) — reusable, mind single-step expiry

- `ExpHst.__init__(N, k)` (lines 4-10): `N` = window size, `k` = `ceil(1/eps)` controls bucket-count invariant. State: `bucket_list` (index 0 = **oldest** surviving bucket), `last` (size of bucket 0), `total` (sum of all bucket sizes).
- `new_bucket(t)` (lines 24-48):
  - Lines 26-28: expiry check — `if bucket_list[0].timestamp < t - window_size: del_last()`. **Only checks the single oldest bucket, once, per call.** This is Finding B: correct only if `t` advances by exactly 1 per call. If two elements arrive at the same logical time, or `t` jumps by more than 1, multiple stale buckets can survive past their expiry undetected. Fix: either guarantee a monotonic +1-per-event logical clock upstream (our chosen approach — see CLAUDE.md §6 Finding B), or change line 26-28 to a `while` loop.
  - Line 30: appends the new element as a fresh `Bucket(t)` of size 1 at the **end** of the list (most recent).
  - Lines 32-48: cascading merge of equal-sized buckets, walking backward (`i` from newest to oldest) counting a run of equal sizes (`rep`); if `rep > ceil(k/2)+1`, merges the **oldest two** buckets in that run into one of double size (`del bucket_list[j+1]` then `set_size(2*s)` on the now-shifted element at the same index — correct, not a bug: after the delete, the second-oldest bucket of the run shifts into slot `j+1`). Loop continues (`i = j+1`) to let the merge cascade upward. This matches the textbook DGIM bucket-invariant maintenance; verified correct on inspection.
- `count_est()` (line 56-57): `total - last/2` — textbook DGIM estimator.
- **Not exercised by the reference at all**: what happens if `new_bucket` is never called before the first `count_est()` on a freshly-constructed `ExpHst` — `total=0, last=0`, returns 0, which is fine. The actual first-arrival bug lives one layer up, in the `*_AKDE.py` callers (see below), not in this file.

## `angular_hash.py` (12 lines) — reusable as-is

`Angular_Hash(dim)` draws one Gaussian normal vector `w ~ N(0,1)^dim` (line 6); `eval(x)` returns `+1` if `w·x >= 0` else `-1` (line 9). This is one bit of random-hyperplane SimHash. `RACE_Ah` concatenates `k` of these and packs the `k` bits into an integer `r` in `[0, 2^k)` (see below) — that integer is the cell index within a row.

## `Ang_hash_AKDE.py` (67 lines) — RE-IMPLEMENT, do not copy; this is where Finding A lives

`RACE_Ah.__init__(rows, hash_range, k, dim, N, eps=0.5)` (lines 6-18):
- `self.R = hash_range` is stored (line 8) but **never read again anywhere in the file** — confirmed dead. The real per-row range is `2^k` (implicit in how `r` is built from `k` bits), not `hash_range`.
- `self.sparse_dic = {}` (line 12): cells are stored sparsely, keyed by `(row_index, hash_code)` rather than a dense `L × 2^k` array — sensible since most cells stay empty.
- `self.hash_list`: `L` rows, each a list of `k` independent `Angular_Hash(dim)` instances.

`update_counter(data, t)` (lines 20-28) — **this is the exact bug location**:
```python
for k,i in enumerate(self.hash_list):
    r=0
    for j in i:
        r=(1 if j.eval(data)==1 else 0)*2+r
    if (k,r) not in self.sparse_dic:
        self.sparse_dic[(k,r)]=ExpHst(self.winsize,math.ceil(1/self.err))   # <-- line 26: create, NO new_bucket() call
    else:
        self.sparse_dic[(k,r)].new_bucket(t)                                # <-- line 28: only the else-branch records the element
```
On a cell's first-ever hit, an empty `ExpHst` is created and the branch stops there — the element that triggered the creation is never recorded via `new_bucket(t)`. Every cell silently drops its first arrival. This is Finding A, and per the paper read on 2026-08-02, it is present in the paper's own Algorithm 2 pseudocode too (§4.1: "if `A[i,j]` is empty then Create an Exponential Histogram... else Add a 1..." — same asymmetry), not just this implementation. **Our port must call the equivalent of `new_bucket(t)` on both branches** (or unconditionally, after ensuring the histogram object exists).

`query1(data)` (lines 30-39): rebuilds each row's cell index the same way, sums `count_est()` across rows that have a cell for this code, divides by `L` — **the mean estimator**, which is what the paper specifies for SW-AKDE (Finding C) and what we use.

`query2(data)` (lines 41-56): median-of-chunk-means (chunk size 5, hardcoded) over the `L` row estimates. **Unused by any driver script in the repo** (confirmed via grep — only `query1` is called anywhere) — vestigial, matches RACE's own median-of-means (for the *un-windowed* RACE_1, not SW-AKDE). Do not port this as the primary query path.

## `RACE_19.py` (68 lines) — baseline, no bug; also the source of `angle_between_vectors`

- `angle_between_vectors(x, y)` (lines 9-26): computes `arccos(clip(x·y / (|x||y|), -1, 1))`. Used as the ground-truth angular distance for brute-force comparisons.
- `RACE_1.update_counter(data)` (lines 42-50): **correct** first-hit handling — `self.sparse_dic[(k,r)] = 1` on creation (line 48), i.e. the triggering element *is* counted immediately. Contrast directly with `Ang_hash_AKDE.py` line 26 above — this is the proof that RACE-proper doesn't have Finding A's bug, only the two EH-augmented sketch files do.
- `query(data)` (lines 52-67): median-of-chunk-means over `L` un-windowed integer counts. This is RACE's real estimator (median-of-means, not mean) — consistent with Theorem 2.3/2.4 in the paper being about plain RACE, and consistent with Finding C (SW-AKDE deliberately switches to mean).
- Use `RACE_1` for **tier 2** validation (§9 in CLAUDE.md): set our SW-AKDE's window ≥ stream length so nothing expires, and confirm our sketch's mean-over-rows estimate and `RACE_1`'s per-cell counts agree in expectation (note the estimator differs — mean vs. median-of-means — so compare against the *counts*, i.e. reconstruct our sketch's un-windowed counter values, not just the final scalar estimate, or compare both against the same brute-force target with enough rows that mean ≈ median-of-means).

## Brute-force ground truth for tier 3 — `compute_true_kde_angular` (defined identically in `mc_error_var.py`, `simulate_AH.py`, `sketch_size.py`, `window_size.py`; used here from `window_size.py` lines 15-25)

```python
def compute_true_kde_angular(data, query, k, window_size):
    window = data[-window_size:]                      # last N points — the actual sliding window
    window_norm = np.linalg.norm(window, axis=1, keepdims=True).clip(min=1e-12)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-12)
    cosines = (query @ window.T) / (query_norm * window_norm.T)
    cosines = np.clip(cosines, -1.0, 1.0)
    angles = np.arccos(cosines)
    kde = np.sum((1.0 - angles / math.pi) ** k, axis=1)   # sum, NOT divided by N
    return kde
```
This is the sketch-independent oracle for angular SW-AKDE: it directly computes `Σ_{x in last-N window} (1 - θ(q,x)/π)^k` — the exact per-row collision probability (`k` = number of concatenated hash bits) summed over the true window, no LSH/sketch involved anywhere. **Note the scale**: this returns a raw sum, not a `1/N`-normalized density. Our sketch's `query1()` mean-over-rows output is on the same unnormalized scale (RACE's `E[A[h(q)]] = Σ k^p(x,q)` per Theorem 2.3 in the paper) — compare directly, do not normalize one side and not the other.

The Euclidean counterpart `compute_true_kde_l2` (same files, e.g. `window_size.py` lines 27-37) uses `l2_lsh_collision_probability` (defined in `L2_hash_AKDE.py` lines 51-56, a closed-form Gaussian p-stable collision probability as a function of Euclidean distance and bucket width `w`) — relevant only when we add the Euclidean kernel in a later pass; angular-only for our Phase 1.

## `L2_hash_AKDE.py` (56 lines) — RE-IMPLEMENT later (angular-only for Phase 1; noted here for completeness)

- Same first-arrival bug, same shape, at lines 34-37 (`if (k,s) not in self.sparse_dic: create ExpHst` / `else: new_bucket(t)`).
- `torch` (line 1) and the `device` line (line 10) are **100% dead** — nothing else in the file references either. Drop both when we port this.
- `self.R = hash_range` **is live** here (line 15, used to construct `UniversalHasher(self.R, ...)` at line 20) — unlike the angular file, don't drop it.
- `p_stable.py` (52 lines, reusable as-is): `PStableLSH(dim, w, p, seed)` draws a Gaussian (`p=2`) or Cauchy (`p=1`) projection vector plus a uniform offset, hashes via `floor((v·a + b)/w)`; `UniversalHasher(R, seed)` folds an arbitrary integer key into `[1, R]` via a Carter–Wegman-style affine hash mod `R`.

## Findings discovered while porting (2026-08-02) — beyond the original audit

**Finding B is deeper than "wall-clock timestamps repeat/skip."** Even with a perfect monotonic per-event logical clock, a single `(row, hash_code)` cell's `ExpHst` only receives `add(t)` calls when *that specific cell* is hit — every element that lands in a different cell/row advances the shared clock without touching this histogram at all. So consecutive calls to one cell's `ExpHst` can be arbitrarily far apart in `t`, and a long-enough gap can leave *more than one* bucket simultaneously expired. The reference's single `if` check (`Exponential_Histogram.py` line 27) only evicts one bucket per call, so it under-evicts in exactly this scenario, independent of whether the driving clock is a logical counter or a wall clock. Verified directly: replaying the same random long-gap sequence, the reference's `if`-based eviction lets `total` drift to **4x** the brute-force true count, while a `while`-loop eviction (evict everything past expiry, not just the oldest bucket) tracks it almost exactly. **Our port (`sketch/exponential_histogram.py`) uses a `while` loop.** This is a strictly more-correct choice CLAUDE.md's Finding B already names as an acceptable alternative ("OR convert EH expiry to a while loop") — we didn't need to pick only one of the two fixes, and did both (logical clock upstream *and* while-loop eviction) since there's no cost to the stronger fix.

**Off-by-one at the window boundary (new, not in the original audit).** The reference's eviction condition (`Exponential_Histogram.py` line 27: `bucket_list[0].get_timestamp() < (t - self.window_size)`) keeps a bucket whose timestamp equals exactly `t - window_size`, giving a window of `{t-N, ..., t}` — **N+1** elements, not N. The paper's own Problem 1.2 defines the window as `T_t = {t-N+1, ..., t}` — exactly N elements. Our port's eviction condition uses `<=` instead of `<` (evict once `timestamp <= t - window_size`) to match the paper's stated semantics exactly. Caught by a tier-1 test that initially failed by exactly one element at the window boundary.

**Finding E — cell code only encodes Hamming weight, destroying LSH amplification (new; affects `Ang_hash_AKDE.py` AND `RACE_19.py`).** Both files build a row's `k`-bit cell code with `r = (bit) * 2 + r` (`Ang_hash_AKDE.py` lines 24/35/48, `RACE_19.py` lines 46/59) instead of the standard shift-and-append `r = r * 2 + bit`. Tracing the recurrence, that collapses to `r = 2 * (number of bits set)` — it records *how many* of the `k` hyperplane hashes fired, not *which ones*. Two points with entirely different bit patterns but equal Hamming weight land in the same cell. Verified empirically (20,000 trials per `k`, fixed cosine similarity 0.5, per-bit collision probability `p ≈ 0.667`):

| `k` | theoretical `p^k` | proper concatenation | reference's actual code |
|---|---|---|---|
| 1 | 0.667 | 0.665 | 0.665 |
| 2 | 0.444 | 0.444 | 0.503 |
| 4 | 0.198 | 0.200 | 0.353 |
| 8 | 0.039 | 0.040 | **0.242** |

Correct at `k=1` (where the two forms coincide up to relabelling: `{0,2}` instead of `{0,1}`, the same partition) and diverges rapidly beyond it — at `k=8` the real collision rate is ~6x theory. This defeats the whole point of concatenating `k` hashes: the amplification that Theorem 2.3/2.4 relies on (`k^p(x,q)` collision probability, range `W^p`) never materialises, so the sketch's effective resolution barely improves as `k` grows.

**Important caveat — this does not invalidate the paper's published results.** The paper states (§5.2) that "the bandwidth parameter, denoted by `p` in Algorithm 2, was set to 1 for all experiments", and every documented invocation in the reference README passes `--b 1`. At `k=1` the bug is inert. It is a *latent* bug: it bites anyone who raises `k` to get the LSH amplification the method is built on — the normal, useful regime — but the reported numbers stand. Say it that way in any writeup; claiming their experiments are wrong would be false.

**Action:** our port uses `code = code * 2 + bit` in both `sketch/sw_akde.py` and `sketch/race.py`, keeping the two an apples-to-apples comparison.

**Finding G — Euclidean cell code sums the k hashes, collapsing the cell space (new; the L2 analogue of E, and worse).** `L2_hash_AKDE.py` lines 30-33 build the cell key as `s = sum(r)`, where `r` holds the `k` universal-hashed p-stable values. Summation is order-invariant *and* range-destroying: it maps the `R^k` distinct hash tuples onto at most `k*R` sums, which by CLT concentrate near the mean, so the usable count is far smaller still. Measured on 4,000 synthetic points, `k=5`, `R = 2^20`:

| | distinct cells | collision rate |
|---|---|---|
| polynomial fold (ours) | 3,384 | 0.154 |
| sum of hashes (reference) | **101** | **0.975** |

Nearly everything lands in ~100 cells regardless of how large `R` is set. Two further notes on the same lines: `s = sum(r)` is computed *inside* the inner loop, so it is recomputed `k` times per row (O(k²) work, only the last value used); and `s` would be undefined if a row had zero hash functions.

Same `k=1` caveat as Finding E — at `k=1`, `sum([h]) == h`, so the reference's own experiments are unaffected.

**Action:** `sketch/p_stable.py`'s `PStableHashBank` folds with a polynomial hash (`code = (code * MULT + value) mod R`), preserving which value came from which position before reducing to the bounded range. `tests/test_sw_akde_euclidean.py::test_fold_preserves_position_unlike_summing` pins this.

**Finding H — the L2 brute-force ground truth omits the `** k` exponent (new).** `compute_true_kde_l2` (`window_size.py` lines 27-37 and its copies) computes `np.sum(probs, axis=1)` where `probs` is the single-hash collision probability — but the collision probability of `k` concatenated hashes is `p^k`, and the angular counterpart in the same file *does* apply `** k` (line 24). So the two ground-truth helpers disagree, and the L2 one is only right at `k=1`. Same caveat: harmless for their `k=1` experiments. **Action:** our `sketch/brute_force.py::compute_true_kde_l2` applies `** k`.

**Finding F — cold cells never expire, so queries return frozen counts (new; the most consequential for our application).** `ExpHst` expires lazily: eviction happens only inside `new_bucket()`. A given cell's histogram is only touched when *that cell* is hit, and `count_est()` (`Exponential_Histogram.py` lines 56-57) takes no time argument at all — it just returns `total - last/2`. So once a cell stops receiving elements, its buckets are never evicted and its reported count is frozen indefinitely, no matter how far the stream has advanced past the window. Demonstrated minimally: add elements at `t=1..5` to a `window_size=10` histogram, then query at any later time — still returns `5.0`, with all five original buckets retained.

This is not merely a code-level omission; the paper's Algorithm 2 query procedure has the same shape (`c ← estimate of count in the Exponential Histogram at A[i, h_i(q)]`, no time parameter), so the reference is faithfully implementing it. But it silently breaks sliding-window semantics in exactly the regime that matters for our use case: **a region going quiet is the anomaly signal**, and without expiry-on-read the density in that region never decays. Caught by a tier-3 test that streamed 300 points into one angular region, then 300 into the antipodal region, and found the original region's density unchanged to the digit (81.365 → 81.365).

**Action:** our `ExponentialHistogram.count_estimate(t)` takes the current logical clock and expires before reading (shared `_expire(t)` helper, also used by `add`); `SlidingWindowAngularKDE.query(x, t)` threads `t` through. Lazy expire-on-read also bounds memory for cold cells, which pure `add`-time eviction does not.

## Hygiene spot-checks (confirmed on disk at the pinned commit)

- The `Code/SlidingWindowKDE/` tree does contain committed `__pycache__/*.pyc` for cpython-38, cpython-310, *and* cpython-313 simultaneously (e.g. `Ang_hash_AKDE.cpython-{38,310,313}.pyc` all present), plus committed `.npy`/`.npz`/`.log`/`.mat` outputs (`synthetic_data/`, `Synthetic_data_outputs/`, `Synthetic_data_outputs_L2/`, `TAKDE/*.pkl`, `data/*.npy`) — confirms §7 of CLAUDE.md. None of this was regenerated or spot-checked for correctness; treat as noise, not ground truth.
- `compare.py` and `compare2.py` are indeed near-duplicates (both define `angle_between_vectors` and near-identical bodies) — confirms the stale-duplicate note.
