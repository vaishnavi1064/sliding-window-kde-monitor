# Project record

Internal reference. The source of truth for what this project is, what it measured, and what went
wrong along the way. Not a résumé, not a paper — completeness and accuracy over polish.

**Rule applied throughout:** every quantitative figure is cited to a committed artifact (code, a
committed CSV, a doc, or a CI run). Nothing is written from memory. Where a number cannot be traced
it is marked `[UNVERIFIED]`. Disagreements between committed documents are recorded in
[§9](#9-internal-disagreements--investigated-and-resolved) — the four found while writing this
document have all been traced to root cause and the source docs corrected.

Repository state at time of writing: 16 commits, branch `phase5-native-core`, HEAD `106d5e8`
(`git rev-list --count HEAD`, `git log`).

---

## 1. Overview

This project takes the sliding-window Approximate KDE (SW-AKDE) sketch from Danait, Das & Bhore,
*Sublinear Sketches for Approximate Nearest Neighbor and Kernel Density Estimation*
([arXiv:2510.23039](https://arxiv.org/abs/2510.23039), Oct 2025) — which composes RACE (Coleman &
Shrivastava, WWW 2020) with Exponential Histograms (Datar, Gionis, Indyk & Motwani, 2002) to give
RACE sliding-window semantics — and does two things the authors did not: **re-implements it
correctly** (their reference implementation, pinned at commit `cc7d7bc1f3902939071a7430a9187be8f05cb237`
on branch `racecms-benchmark`, contains defects that our port fixes, some of which trace to the
paper's own pseudocode), and **applies it to real industrial sensor data** (MetroPT-3 air-production
unit) to establish where the method actually pays off. The algorithm is not ours and no novelty is
claimed for it. The contributions are engineering (a correct, tested, memory-bounded port; a C++17
core that is bitwise-identical to it and ~10–20× faster) and a **negative applicability result**: at
MetroPT's seven channels the sketch costs 26–236× *more* memory than simply storing the window
(`docs/EVALUATION.md` §4 — and worse still on a peak-footprint basis, §9.1), so this pairing sits
outside the regime where the algorithm's central advantage exists. Reporting that boundary, measured,
is the useful output.

---

## 2. Phase-by-phase record

Each phase: what was built, its definition of done, and what went wrong or was discovered — the last
being the point of this section. Phase definitions and DoDs are from `CLAUDE.md` §12.

### Phase 0 — Setup

**Built.** Clean repository separate from the reference tree; `CLAUDE.md` as the project brief;
`docs/REFERENCE_NOTES.md` carrying the line-level audit. Reference kept as read-only material, pinned
to commit `cc7d7bc1f…`, archived outside the repo.
**DoD.** Clean repo, brief at root, notes in `docs/`.
**Discovered.** The reference's useful code is **only on the non-default branch** `racecms-benchmark`;
`main` is missing the entire `Code/SlidingWindowKDE/` directory (`CLAUDE.md` §4). Also that ~86% of
tracked files in that directory are committed artifacts, including `__pycache__` bytecode for CPython
3.8, 3.10 **and** 3.13 simultaneously, contradicting the README's stated "Python 3.12.0"
(`CLAUDE.md` §7; confirmed on disk, `docs/REFERENCE_NOTES.md` line 124). Consequence adopted:
committed numeric outputs in the reference are **not** treated as authoritative.

*Committed as* `9ba0349` (with Phase 1).

### Phase 1 — Validated pure-Python port

**Built.** `sketch/` as an importable package: `exponential_histogram.py`, `buckets.py`,
`angular_hash.py`, `p_stable.py`, `sw_akde.py`, `race.py` (un-windowed baseline), `brute_force.py`
(sketch-independent oracle). Three validation tiers per `CLAUDE.md` §9.
**DoD.** Windowed KDE matches brute-force truth within tolerance; `pytest` green; parity with
`RACE_19` in the un-windowed limit. Hard gate before any optimization.
**Tolerance used.** Mean relative error ≤ `2·eps + eps²` = **0.21** at `eps = 0.1`
(`tests/test_sw_akde_windowed.py` lines 7–11, citing the paper's Lemma 4.3 — the same figure the
paper quotes for its own experiments, §5.2).

**What went wrong / was discovered.**

- **Findings A and F were found here, and both trace to the paper's Algorithm 2, not just the
  reference code.** Finding A (every cell drops the element that created it) and Finding F (cold
  cells never expire, so density never decays). Details and validation in [§4](#4-correctness-findings).
- **Findings E, G, H were found here too — and the initial framing of them was an overclaim that
  had to be corrected.** The first reading was that these bugs undermined the paper's results. They
  do not: the paper sets the concatenation parameter to 1 for all experiments (§5.2) and every
  documented reference invocation passes `--b 1`, at which all three are **inert**
  (`docs/REFERENCE_NOTES.md` line 97). They are *latent* — they break as soon as `k > 1`, which is
  the regime the whole LSH-amplification argument is about. The corrected wording is explicit that
  "claiming their experiments are wrong would be false" (`docs/REFERENCE_NOTES.md` line 97), and
  `CLAUDE.md` §6 carries the same caveat. **How it was caught:** by reading the paper's §5.2 and the
  reference README's own invocations rather than inferring impact from the code alone. This is the
  single most important framing correction in the project.
- **Two smaller EH defects surfaced from tier-1 tests**, not from reading: multi-bucket eviction
  (a single `if` under-evicts; verified to let `total` drift to **4×** the brute-force count,
  `docs/REFERENCE_NOTES.md` line 82) and a **window-boundary off-by-one** (`<` keeps `N+1` elements
  where the paper's Problem 1.2 defines `N`; "caught by a tier-1 test that initially failed by
  exactly one element", `docs/REFERENCE_NOTES.md` line 84).
- **Finding C was investigated and turned out not to be a defect.** Mean vs median-of-means is not a
  code/paper divergence: the paper specifies the mean for SW-AKDE, and the reference's `query2` is
  vestigial and unused (`CLAUDE.md` §6, confirmed by reading §4.1 directly).

### Phase 1.5 — Euclidean kernel and the measured performance baseline

**Built.** `PStableHashBank` (Euclidean/p-stable LSH with a polynomial fold), vectorized
`AngularHashBank`, and a profile-driven optimization pass.
**DoD.** Not a `CLAUDE.md` phase; an interposed engineering step so that any later C++ comparison
would be against an *optimized* Python baseline rather than a naive one.

**What was discovered.** Profiling first, then optimizing — the numbers are in
`docs/PERFORMANCE.md` §§1–3:

- Before: **15,180,865 function calls in 6.646 s** over 2000 updates at `rows=400`; split **hashing
  ~59%**, exponential histogram ~31%. The cost was *call overhead*, not arithmetic: each update
  issued `rows × k` = 2000 separate `np.dot` calls on 15-element vectors. `math.ceil` appeared
  **812,800** times recomputing a constant.
- After: **5,599,665 calls in 2.319 s** — a **2.9×** speedup, test suite 14.4 s → 3.5 s.
- Hashing dropped out of the profile entirely; the exponential histogram became **~72%** of update
  time. This measurement is what later determined Phase 5's target — the EH and cell array, *not*
  the hashing (`docs/PERFORMANCE.md` line 114).
- `docs/PERFORMANCE.md` line 119 records the constraint this creates: the C++ benchmark must be
  quoted against *this* baseline, and folding the 2.9× into a native-core figure "would be
  dishonest."

*Committed as* `7592ac7`, `9cff110`.

### Phase 2 — Streaming pipeline

**Built.** Kafka (KRaft, single broker) via `docker compose`; producer replaying MetroPT-3 with a
**logical clock**; consumer owning the clock, updating the sketch, computing rolling anomaly and
data-quality scores; Prometheus metrics; a Grafana dashboard; an Alertmanager rule.
**DoD.** `docker compose up` → live dashboard reacting to the stream, alert fires on an injected
anomaly.

**What went wrong — four things the real data forced** (`docs/DATA_NOTES.md`, all four changed code):

1. **MetroPT-3 is 0.1 Hz, not 1 Hz.** Both the dataset documentation and our own brief said 1 Hz.
   Measured across all 1,516,948 readings: median inter-reading gap **10.0 s**, mean **12.1 s**,
   p1/p25/p75/p95/p99 = **9 / 10 / 10 / 10 / 12 s**, **0** duplicate timestamps, **0** timestamp
   regressions (`docs/DATA_NOTES.md` §1). The gap threshold was 5 s, so **100% of readings were
   flagged as gaps** and the data-quality alert sat permanently firing; now 30 s with a regression
   test pinning the 10 s cadence as clean. Also: `window_size = 3600` is **~10 hours** of asset
   time, not one hour.
   - Honest downgrade recorded in the same section: because there are 0 duplicate and 0 backwards
     timestamps, the wall-clock hazard behind Finding B "is milder here than feared." The logical
     clock is still used, but the danger is explicitly not overstated.
2. **The first detector did not work, and more smoothing could not fix it.** Normal density spans
   nearly three orders of magnitude (median **81.2**, IQR **39.1–140.7**, range **1.0–528.4**;
   injected-fault median **9.3**) — a **9×** median drop, yet a robust z-score of raw density peaked
   at **1.4** against a threshold of 3, with a sweep ceiling of **1.41**
   (`docs/DATA_NOTES.md` §2, from `scripts/tune_detector.py`, 2,441 density samples). Cause: density
   is positive and multiplicative, so on a linear scale a 9× collapse is unremarkable. Fix: score in
   **log space** (`log1p`), which took the same fault from 1.41 to **4.13**.
3. **Optimising peak height was the wrong target.** The smoothing sweep (`docs/DATA_NOTES.md` §2)
   shows span 10 gives the tallest peak (4.13) but only **5 s** above threshold, while span **80**
   gives a lower peak (**2.44**) with **87 s** above threshold and the best separation (**3.5×**,
   worst normal **0.70**). Thresholds follow from that table — warning **1.3**, critical **2.0** —
   not from taste.
4. **Density is not comparable until the window has filled**, and a sliding-window density signal is
   inherently **transient** (`docs/DATA_NOTES.md` §§3–4). Feeding the fill-up ramp to the scorer
   poisons its baseline; the consumer now gates scoring on `clock >= window_size`.

Method note recorded in `docs/DATA_NOTES.md`: the first two detector attempts were tuned by
intuition against slow end-to-end Docker runs and both were wrong; moving to an offline sweep
against the real sketch exposed both immediately.

*Committed as* `1425c69`.

### Phase 3 — Detection and evaluation (the applicability finding)

**Built.** Labeled detector; per-event evaluation against the four documented failures; baselines
(un-windowed RACE, exact windowed KDE); MLflow tracking; figures with a CSV beside each.
**DoD.** Evaluation report with metrics and plots vs baselines, logged to MLflow.

**What went wrong / was discovered — three findings, two of them unwelcome.**

1. **Cell reclamation: the sketch leaked memory by design.** Found by running the full dataset
   rather than a benchmark — throughput decayed from **~810 to ~300 updates/s** part way through
   1.5M readings while the process reached **2.7 GB** (`docs/PERFORMANCE.md` §"Cell reclamation").
   Not an ordinary leak: the published space bound `O(RW/eps · log²N)` counts a **dense** `R × W`
   array where an unused cell is free, but every real implementation stores cells **sparsely**, and
   nothing then bounds the dictionary — it gains one entry per distinct cell ever visited, so memory
   tracks *readings seen* rather than *window size*. This is the memory half of Finding F. Fix:
   `compact()` drops fully expired cells once per window, which is semantically free.
2. **The detector detects at onset but does not predict — and the naive reading of its own results
   was wrong.** See [§5](#5-the-negative-result). The chance calibration is what changed the reading;
   without it, "detects all four failures with 17 hours of lead time" would have been reported and
   would have been badly misleading (`docs/EVALUATION.md` §"Calibrating against luck").
3. **The applicability boundary — the main reportable finding.** At `dim=7` the sketch costs
   **26–236×** more memory than storing the window. See [§3](#3-every-verified-number-with-source)
   and `docs/EVALUATION.md` §4. Reported as the contribution rather than engineered around; the doc
   explicitly recommends exact windowed KDE over our own sketch for MetroPT.

Two further honesty items recorded in the same phase:

- **A feature-engineering attempt failed and is reported as such.** Adding the eight digital
  channels' rolling duty cycles made things **worse**: best p went from **0.108 to 0.193**
  (`docs/EVALUATION.md` §3) — the ordinary curse of dimensionality for KDE, going from 7 to 15
  dimensions. Also flagged there: those features were chosen *after* inspecting the four events,
  which with n=4 is a real overfitting risk.
- **TAKDE was dropped, with reasons.** `CLAUDE.md` listed it as a bonus baseline; on reading
  `TAKDE/takde_alg.py` it is a **1-D synthetic benchmark script**, not a reusable implementation
  (window selection via 1-D `np.histogram`, ground truth assumed Gaussian, error computed inline
  against that assumption). Running it on 7-D data requires choosing a multivariate two-sample
  statistic the method's reference does not provide, so any weakness would reflect our choice rather
  than the published method (`docs/EVALUATION.md` §"TAKDE was considered and dropped").

*Committed as* `14c7887`, `859836d`, `531ba86`, `f16976d`, `3b6a45d`, plus `8c8df4b` — a README
verification pass whose subject line is itself part of this record: *"fix stale numbers, evidence
Findings A and F."*

### Phase 4 — MCP server

**Built.** MCP server over Postgres-backed alert history exposing `get_asset_health`,
`list_recent_anomalies`, `explain_alert`; `scripts/load_alerts.py` to load detected episodes;
`scripts/verify_mcp.py` to exercise the tools end to end.
**DoD.** An MCP client queries health and recent anomalies end to end.
**What went wrong.** Dev credentials were initially inline and were moved to environment variables
(`5dc5530`, *"Verify Phase 4 end to end, and move dev credentials to environment variables"*);
`.env` is gitignored with only `.env.example` tracked.
**Latent defect introduced here, found much later (Phase 5 CI work):** `mcp_server/server.py` line
30 imports `mcp` unguarded at module scope, which is what made the second CI bug invisible until the
first was fixed — see below.

*Committed as* `a5106cb`, `5dc5530`.

### Phase 5 — Native C++17 core

**Built.** `src/exponential_histogram.hpp`, `src/cell_array.hpp`, `src/bindings.cpp` — a C++17 cell
array behind pybind11, exposed as `sketch._native` and selected via `backend="native"`. Hashing
deliberately stays in NumPy (settled by the Phase 1.5 profile). One pybind11 crossing per update,
GIL released for the row loop. A cell-store seam (`sketch/cell_store.py`) lets both cores sit behind
one interface; the Python core remains the **default and the oracle** (`CLAUDE.md` §13).
**DoD.** Benchmark report showing speedup + identical accuracy.

Two representational changes carry the win (`docs/PERFORMANCE.md` §"Phase 5"): a `Bucket` is **16
bytes of POD** rather than a **56-byte** `PyObject` plus an **8-byte** list slot, and buckets live in
a power-of-two ring buffer so eviction is a head-index bump instead of `list.pop(0)`.

**What went wrong — five incidents, all caught.**

1. **Four of five headline benchmark numbers were inflated, and were caught only by a clean
   re-run.** The first benchmark run produced 11–22× update and 6–18× query speedups, a 19×
   pipeline figure and a "~77 s" full pass. A clean re-run from a scratch build contradicted four of
   the five. Both runs are now recorded side by side in `docs/PERFORMANCE.md`
   §"Run-to-run variance is real and worth stating": re-run **10.3–19.0×** updates and **6.1–13.9×**
   queries vs the earlier **10.9–22.1×** and **6.0–17.5×** — individual per-row ratios move by up to
   **~3×** between runs on a non-isolated desktop. The doc now quotes ranges (**~10–20×** updates,
   **~6–14×** queries) rather than any single table cell, and distinguishes timed columns from
   structural ones. **This is the second time in the project that a published number did not survive
   re-measurement** (the first being the Phase 3 README pass, `8c8df4b`), which is why this document
   exists.
2. **A stated cause for the p99 latency tail was wrong, and testing it flipped the conclusion.** The
   plausible story — compaction runs 1 update in 256 (0.4%), landing exactly inside the top 1% — was
   checked rather than assumed. Removing compaction moves p99 by **5%** and in the *opposite*
   direction (92.2 µs with vs 97.1 µs without at `rows=400`), and the ordering of the extreme values
   **flips between runs** (`docs/PERFORMANCE.md` §"On the p99 tail"). The tail is scheduling and
   allocator jitter. An earlier draft's claim that "the maximum is higher without compaction" was a
   one-run artifact and was removed.
3. **"Phase 3's finding is unaffected" was wrong and was corrected.** The crossover is *linear in*
   `bytes_per_cell`, so a 2.2× leaner cell shifts it by exactly that factor: the rule of thumb moves
   from `dim > ~5 × rows` to `dim > ~2.3 × rows` (`docs/EVALUATION.md` lines 279–286,
   `docs/PERFORMANCE.md` §"What is still true"). The *conclusion* is unchanged — seven channels is
   still two orders of magnitude short — but "unaffected" was the wrong word for a constant-factor
   win on a boundary that depends linearly on that constant.
4. **Two documented numbers had no committed backing.** The p99-tail comparison and the
   evaluation-parity throughput came from throwaway probe scripts. Both were folded into
   `scripts/benchmark_native.py` as `--tail-check` and `--eval-parity`, so every Phase 5 figure now
   regenerates from committed code.
5. **Two stacked CI bugs, the second hidden behind the first.** CI had **never** been green in this
   repository. `pytest` aborted during collection because `tests/test_evaluation.py` and
   `tests/test_mcp_server.py` import pandas at module scope while CI installed only `pytest`. The
   obvious one-line fix — installing `.[dev,streaming]`, since pandas lives in the `streaming`
   extra — **would not have worked**: pandas was masking `mcp_server/server.py` line 30's unguarded
   `from mcp.server.mcpserver import MCPServer`, and `mcp` is in neither extra. Reproduced in a
   clean venv before changing anything. Fixed by putting both packages in the `dev` extra rather
   than widening the CI install line, which would have pulled confluent-kafka, pyarrow, requests,
   prometheus-client and psycopg[binary] that no test touches (`psycopg` is imported lazily inside
   `mcp_server/store.py` line 102). Fixed in `35f66ba`; the defect predates Phase 5 and was failing
   identically on `main` at `a5106cb` and `5dc5530`.
6. **The container path was verified only after being flagged unverified through three
   commits.** `SKETCH_BACKEND` defaults to `auto`, and `python:3.12-slim` has no compiler — so an
   image built without one would silently run the Python core while the docs advertised a ~20×
   figure. The Dockerfile installs g++, builds, purges the compiler, and *then* asserts the module
   imports (that ordering is deliberate: an `autoremove` that took `libstdc++6` fails the build
   rather than shipping an image that dies on start). Verification is in [§7](#7-verification-state).

*Committed as* `4ab5ef8`, `6c7c6fc`, `35f66ba`, `106d5e8`.

### Phase 6 — Adaptive window size

**Not attempted.** `CLAUDE.md` §12 lists it as an optional stretch (Lever 3) contingent on Levers
1–2 being solid; `README.md` status table marks it "Stretch."

---

## 3. Every verified number, with source

`S` = **structural** (deterministic; reproduces exactly). `M` = **measured** (timing/OS-dependent;
varies between runs). Read `M` figures as ranges.

### 3.1 Phase 5 — native vs Python core

| Metric | Value | Kind | Source |
|---|---|---|---|
| Cell memory, Python core | 63.9 MB | S | `docs/PERFORMANCE.md` §Memory table |
| Cell memory, native core | 29.0 MB | S | same |
| Cell-memory reduction | **2.2×** | S | same; "reproduced exactly across both runs … to the byte" |
| Bytes per cell, Python | 761 | S | same — at this clock; the figure is compaction-phase dependent (§9.1) |
| Bytes per cell, native | 345 | S | same |
| Bytes-per-cell **ratio** 345/761 | **0.453** | S | phase-matched (both cores, same clock, identical cell count) — this is what the crossover correction rests on |
| Live cells, both cores | **83,920 (identical)** | S | same — identical counts are the parity check |
| RSS delta, Python / native | 113.9 MB / 54.8 MB → 2.1× | M | same; independent OS-level corroboration of the 2.2× |
| Update speedup (defensible range) | **~10–20×** | M | `docs/PERFORMANCE.md` §Throughput; re-run 10.3–19.0×, earlier run 10.9–22.1× |
| Query speedup (defensible range) | **~6–14×** | M | same; re-run 6.1–13.9×, earlier run 6.0–17.5× |
| Per-row variance between runs | up to ~3× | M | same, §"Run-to-run variance" |
| Pipeline throughput, Python / native | 1,060 / 22,659 upd/s → **21.4×** | M | `docs/PERFORMANCE.md` §Memory table; doc says "treat it as ~20x" (earlier run gave 19.0×) |
| Full MetroPT-3 pass, Python / native | **23.9 min / 1.1 min** | M | same table, `full pass` column (1,516,948 readings) |
| Native p50 / p99 latency, `rows=400` | 32.8 µs / 92.0 µs | M | `docs/PERFORMANCE.md` §Throughput table |
| Hash share of native update time | 45.2% at `rows=100` → 15.1% at `rows=3200` | M | same table — the Amdahl ceiling on this approach |
| p99 with / without compaction | 92.2 µs / 97.1 µs (5%, opposite direction) | M | `docs/PERFORMANCE.md` §"On the p99 tail"; `--tail-check` |
| End-to-end eval parity throughput | 1,132 / 15,356 readings/s (13.6×) | M | `docs/PERFORMANCE.md` §Accuracy; `--eval-parity --records 30000` |
| End-to-end eval parity result | `timestamp`, `density`, `score` **identical** | S | same; `pandas.Series.equals`, not a tolerance |

Reproduce: `python -m scripts.benchmark_native` (plus `--tail-check`, `--eval-parity`), listed in
`docs/PERFORMANCE.md` §"Phase 5". Note `make` is **not installed** on the development machine, so
these ran as module invocations; `README.md` §"Without `make`" carries the equivalence table.

### 3.2 Phase 1.5 — Python optimization baseline

| Metric | Value | Kind | Source |
|---|---|---|---|
| Before / after optimization | 6.646 s / 2.319 s = **2.9×** | M | `docs/PERFORMANCE.md` §§1,3 (cProfile, 2000 updates, `rows=400`) |
| Function calls before / after | 15,180,865 / 5,599,665 | S | same |
| Hashing share before | ~59% | M | `docs/PERFORMANCE.md` §1 |
| EH share after | ~72% | M | `docs/PERFORMANCE.md` §3 — determined Phase 5's target |
| Redundant `math.ceil` calls | 812,800 | S | `docs/PERFORMANCE.md` §1 |
| Test suite before / after | 14.4 s / 3.5 s | M | `docs/PERFORMANCE.md` §3 |
| Python throughput, `rows=100 … 3200` | 9,909 → 112 upd/s | M | `docs/PERFORMANCE.md` §Throughput (`make bench`) |

### 3.3 Cell reclamation (Phase 3)

Measured over 400,000 MetroPT-3 readings, `rows=400`, `window=3600` (`docs/PERFORMANCE.md`
§"Cell reclamation", `make memcheck`):

| Metric | Without compaction | With compaction | Ratio | Kind |
|---|---:|---:|---:|---|
| Live cells | 1,859,317 | 156,644 | 11.9× fewer | S |
| Cell memory | 666 MB | 83 MB | 8.0× smaller | S |
| Throughput | 794/s | 922/s | 1.2× faster | M |
| Deceleration | 1.34× | 1.18× | — | M |

Also: **4.8M** dead cells reclaimed over the run; the un-compacted full-dataset run reached
**2.7 GB** with throughput decaying **~810 → ~300 upd/s** (same section).

### 3.4 The applicability boundary

`window=3600`, `dim=7` (`docs/EVALUATION.md` §4, `make memcheck-rows`):

| rows | live cells | sketch | vs exact | crossover dim | Kind |
|---:|---:|---:|---:|---:|---|
| 50 | 7,398 | 5.2 MB | 26× | 179 | S |
| 100 | 15,383 | 12.0 MB | 60× | 418 | S |
| 200 | 29,849 | 23.6 MB | 117× | 819 | S |
| 400 | 60,330 | 47.5 MB | 236× | 1,650 | S |

**Re-run and confirmed:** `scripts/memory_check.py --records 150000 --rows-sweep` reproduces every
cell of this table exactly (2026-08-05). Exact windowed KDE holds `3600 × 7` floats = **0.20 MB**; the
sketch needs 5–48 MB for the same detection quality — **26–236× more expensive than the thing it
replaces**. `docs/figures/memory_crossover.csv` carries `crossover_dim` 180.56 for `rows=50`; that is
a rounding artifact, not a better number — see §9.2.

**These are near-trough samples.** The live cell count sawtooths ~2× within each compaction cycle, so
peak footprint is roughly **1.6×** these figures (§9.1) — ~385× exact storage at `rows=400` rather
than 236×.

| Metric | Value | Source |
|---|---|---|
| `bytes_per_cell` (doc: "constant") | ~590–800 bytes — actually the sawtooth span, see §9.1 | `docs/EVALUATION.md` §"The boundary, as a rule" |
| `bytes_per_cell` observed over one cycle | 663 (peak cells) … 802 (post-sweep) | §9.1 probe, `rows=400` |
| `cells_per_row` vs window | 80 / 120 / 206 / 242 at window 900 / 1800 / 3600 / 7200 | same |
| crossover ÷ rows at those windows | 6.7 / 5.5 / 5.0 / 3.4 | same |
| `cells_per_row` growth | ≈ `window^0.55` | same |
| **Rule, Python core** | `dim > ~5 × rows` | `docs/EVALUATION.md` line 277 |
| **Rule, native core (corrected)** | `dim > ~2.3 × rows`, i.e. ~750 dim at `rows=400` | `docs/EVALUATION.md` lines 279–286 (scales by 345/761) |
| Paper's own experiment dimensions | 103, 200, 384 | `docs/EVALUATION.md` §"Why that is an awkward rule" — inside the useful regime |

### 3.5 Detection results

Full 1,516,948-reading replay, `rows=400`, `k=3`, `window=3600`, 7 analog channels. Config constants
verified from `streaming/config.py` and `streaming/features.py` (`rows=400`, `k=3`, `window=3600`,
`width=2.0`, `eh_relative_error=0.1`, `warmup=2000`, 7 `ANALOG_COLUMNS`).

Exact values from `docs/figures/operating_points.csv` (committed):

| Method | Alerting | Events | p | False alarms/day | Mean lead (h) |
|---|---:|---:|---:|---:|---:|
| swakde | 20% | 4/4 | 0.5665 | 1.291 | 16.79 |
| swakde | 10% | 4/4 | 0.2905 | 0.784 | 13.70 |
| **swakde** | **5%** | **4/4** | **0.0495** | **0.346** | **10.06** |
| swakde | 2% | 2/4 | 0.1935 | 0.112 | 10.72 |
| swakde | 1% | 2/4 | 0.0175 | 0.019 | −1.06 |
| exact | 5% | 4/4 | 0.0490 | 0.341 | 10.07 |
| race | 5% | **3/4** | 0.2845 | 0.365 | −1.38 |

So the sketch matches exact windowed KDE to within noise (**0.0495 vs 0.0490** at the 5% budget) —
approximation costs essentially nothing in detection quality. Un-windowed RACE is worse where it
matters (3/4 at 5%), which supports the paper's sliding-window claim.

Horizon sensitivity (`docs/EVALUATION.md` §2):

| Horizon | Events | Mean lead | p | Per-event lead |
|---|---:|---:|---:|---|
| **3 h** | **4/4** | **−0.6 h** | **0.004** | −2, −0, −0, −0 |
| 6 h | 4/4 | −0.6 h | 0.009 | −2, −0, −0, −0 |
| 12 h | 4/4 | −0.6 h | 0.024 | −2, −0, −0, −0 |
| 24 h | 4/4 | +10.1 h | 0.049 | +19, +22, −0, −0 |
| 48 h | 4/4 | +27.2 h | 0.135 | +19, +43, +47, −0 |

Chance calibration: **2,000** random trials per operating point (`docs/EVALUATION.md` line 63).
The control that mattered most: un-windowed RACE at a fixed threshold of 1.3 detects **4/4** — while
alerting **17.1%** of the time, with chance alone catching **3.21/4** and **p = 0.399**
(`docs/EVALUATION.md` §"Calibrating against luck").

Per-failure signal, `Oil_temperature` standardized shift in the 24 h before onset, exact values from
`docs/figures/effect_sizes.csv`:

| Failure | Oil_temperature | Readings | Verdict (`docs/EVALUATION.md` §3) |
|---|---:|---:|---|
| failure-1 | **−1.906σ** | 6,301 | clearly visible |
| failure-2 | **+0.622σ** | 7,466 | barely visible |
| failure-3 | **+0.235σ** | 8,716 | **not visible** |
| failure-4 | **+1.668σ** | 6,185 | clearly visible |

Peak scores around each failure, from `docs/figures/scores_around_failures.csv` (before / after
onset): failure-1 3.183 / 3.933; failure-2 4.710 / 6.955; failure-3 4.867 / 10.423; failure-4
2.600 / 2.943 — the score peaks *after* onset in all four.

Digital-channel duty-cycle attempt: best p **0.108 → 0.193** (worse); underlying digital shifts
`Oil_level` 0.901 → 1.000, `Caudal_impulses` 0.935 → 1.000, `DV_eletric` 0.143 → 0.024 or 0.393
(`docs/EVALUATION.md` §3).

### 3.6 Detector tuning (Phase 2)

From `docs/DATA_NOTES.md` §2, `scripts/tune_detector.py`, 2,441 density samples, 30,000 readings
with a synthetic 4× fault:

| Metric | Value |
|---|---|
| Normal density: median / IQR / range | 81.2 / 39.1–140.7 / **1.0–528.4** |
| Injected-fault density: median / IQR / range | 9.3 / 5.0–18.2 / 1.0–62.2 |
| Median drop under fault | **9×** |
| Linear-scale z-score peak (threshold 3) | **1.4**; sweep ceiling **1.41** |
| Same fault, log space, short smoothing | **4.13** |
| Chosen smoothing (EWMA span 80) | peak 2.44, worst normal 0.70, separation **3.5×**, 87 s above threshold |
| Thresholds | warning **1.3**, critical **2.0** |
| Fault contamination of rolling history | ~17–18% (motivates median + MAD) |

### 3.7 Test and CI counts

| Metric | Value | Source |
|---|---|---|
| Tests with the native extension | **130 passed** | CI run 30981570447, both native jobs |
| Tests without it | **93 passed, 37 skipped** | CI run 30981570447, `python-core` job |
| Parity tests executed in native jobs | **37, 0 skipped** | same |
| Parity tests in pure-Python job | 0 executed, **37 skipped** | same — the control |
| Total commits | 16 | `git rev-list --count HEAD` |

---

## 4. Correctness findings

Findings A–H from `CLAUDE.md` §6 and `docs/REFERENCE_NOTES.md`. **The A/F vs E/G/H distinction is
the one to keep exact.**

### In the published algorithm (affect Algorithm 2 itself, active at the paper's own `k=1`)

**Finding A — every cell silently drops its first arrival.** Algorithm 2's preprocessing reads
`if A[i,j] is empty then Create an Exponential Histogram … else Add a 1 …` — the create branch has no
corresponding "Add a 1", so the element that created the cell is never counted. The reference repeats
it (`Ang_hash_AKDE.py` lines 25–28); `RACE_19.py` line 48 does **not** have it, which is the proof
that only the two EH-augmented files are affected (`docs/REFERENCE_NOTES.md` line 52). Impact is
concentrated in the sparse-cell regime. **Validated by** `test_every_cells_first_arrival_is_counted`:
a single element gives density **1.0** when counted correctly and exactly **0.0** under the published
branch structure. The tier-2 convergence test cannot see it — a one-per-cell undercount measures
**0.074 vs 0.063** mean relative error, both inside its 0.30 tolerance (`README.md` §Correctness),
which is why it is tested separately. **Consequence adopted:** never validate against the reference's
sketch outputs (Finding D).

**Finding F — cold cells never expire, so density is frozen forever.** `count_est()`
(`Exponential_Histogram.py` lines 56–57) takes no time argument, and Algorithm 2's query procedure
has the same shape (`c ← estimate of count in the Exponential Histogram at A[i, h_i(q)]`), so the
reference is faithfully implementing the paper. **The most consequential finding for this
application:** a region going quiet *is* the anomaly signal, and without expiry-on-read that density
never decays. **Validated two ways:** minimally, adding elements at `t=1..5` to a `window_size=10`
histogram still returns **5.0** at any later time (`docs/REFERENCE_NOTES.md` line 116); and at
sketch level, streaming 300 points into one angular region then 300 into the antipodal region left
the first region's density **unchanged to the digit (81.365 → 81.365)** (line 118). Our
`count_estimate(t)` expires before reading. Also has a memory half — see §3.3.

### Implementation defects in the reference, latent at `k=1`

`docs/REFERENCE_NOTES.md` line 97 and `CLAUDE.md` §6 both state the caveat explicitly: the paper sets
the concatenation parameter to 1 for all experiments (§5.2) and every documented invocation passes
`--b 1`, so **all three are inert there and the paper's published numbers stand.** They break at
`k > 1` — the regime the LSH-amplification argument is about.

**Finding E — the angular cell code encodes only Hamming weight.** Both `Ang_hash_AKDE.py` and
`RACE_19.py` build the code as `r = bit*2 + r` instead of `r = r*2 + bit`, which collapses to
`r = 2 × (number of bits set)`: *which* hashes fired is lost. **Validated** over 20,000 trials per
`k` at fixed cosine similarity 0.5 (per-bit `p ≈ 0.667`), `docs/REFERENCE_NOTES.md` line 88:

| `k` | theoretical `p^k` | proper concatenation | reference |
|---:|---:|---:|---:|
| 1 | 0.667 | 0.665 | 0.665 |
| 2 | 0.444 | 0.444 | 0.503 |
| 4 | 0.198 | 0.200 | 0.353 |
| 8 | 0.039 | 0.040 | **0.242** |

At `k=8` the real collision rate is **~6×** theory. Correct at `k=1` up to relabelling.

**Finding G — the Euclidean cell code sums the `k` hashes** (`L2_hash_AKDE.py` lines 30–33), which is
order-invariant and range-destroying. **Validated** on 4,000 points, `k=5`, `R = 2^20`
(`docs/REFERENCE_NOTES.md` line 103): polynomial fold (ours) **3,384** distinct cells, collision rate
**0.154**; sum of hashes (reference) **101** cells, collision rate **0.975**. Pinned by
`tests/test_sw_akde_euclidean.py::test_fold_preserves_position_unlike_summing`.

**Finding H — the L2 brute-force ground truth omits `** k`.** `compute_true_kde_l2` sums the
single-hash collision probability without raising it to `k`, while the angular helper in the same
file does — so the two oracles disagree and the L2 one is right only at `k=1`
(`docs/REFERENCE_NOTES.md` line 114). Our `compute_true_kde_l2` applies `** k`.

### Sliding-window semantics defects

**Finding B — single-step expiry.** Deeper than "wall-clock timestamps repeat": even with a perfect
logical clock, one cell's histogram only receives `add(t)` when *that cell* is hit, so consecutive
calls can be arbitrarily far apart and more than one bucket can be expired at once. The reference's
single `if` (`Exponential_Histogram.py` line 27) evicts one bucket per call. **Validated:** replaying
the same random long-gap sequence, the `if`-based eviction lets `total` drift to **4×** the
brute-force count, while a `while` loop tracks it almost exactly (`docs/REFERENCE_NOTES.md` line 82).
We applied *both* available fixes — a logical clock upstream and while-loop eviction.

**Window-boundary off-by-one** (not in the original audit). The reference's `<` keeps a bucket whose
timestamp equals `t − window_size`, giving `{t−N, …, t}` = **N+1** elements where the paper's
Problem 1.2 defines `T_t = {t−N+1, …, t}` = N. Ours uses `<=`. **Caught by a tier-1 test that
initially failed by exactly one element at the boundary** (`docs/REFERENCE_NOTES.md` line 84).

### Not defects

**Finding C — the mean estimator is correct, not a divergence.** The paper specifies the mean for
SW-AKDE ("the average of ACE estimates over L independent repetitions", §4.1); median-of-means is
plain RACE's estimator. The reference's `query2` is vestigial and unused (`CLAUDE.md` §6).

**Finding D — validation strategy, not a bug.** Validate against the sketch-independent brute-force
KDE the reference already contains, never against `*_AKDE.py` outputs (`CLAUDE.md` §6).

### The count is seven

`README.md` and `docs/EVALUATION.md` previously said "eight real bugs/defects." That was never
supported by the enumeration and has been corrected to **seven** in both places. The seven, in full:

| # | Defect | Where it lives | Active at the paper's `k=1`? |
|---:|---|---|---|
| 1 | **A** — every cell drops its first arrival | Algorithm 2 **and** `Ang_hash_AKDE.py` 25–28 | **Yes** |
| 2 | **F** — cold cells never expire; density frozen | Algorithm 2 **and** `Exponential_Histogram.py` 56–57 | **Yes** |
| 3 | **B** — single-step expiry under-evicts | `Exponential_Histogram.py` 27 | Yes (drives `total` to 4× truth) |
| 4 | **Window-boundary off-by-one** — `<` keeps `N+1` elements | `Exponential_Histogram.py` 27 | Yes |
| 5 | **E** — angular cell code encodes only Hamming weight | `Ang_hash_AKDE.py`, `RACE_19.py` | No — latent, bites at `k>1` |
| 6 | **G** — Euclidean cell code sums the `k` hashes | `L2_hash_AKDE.py` 30–33 | No — latent, bites at `k>1` |
| 7 | **H** — L2 ground truth omits `** k` | `window_size.py` 27–37 and copies | No — latent, bites at `k>1` |

**C and D are excluded because they are not defects.** C (mean vs median-of-means) is the reference
being *correct* — the paper specifies the mean for SW-AKDE. D is our validation strategy, not a bug
in the source material. Any count that reaches eight is double-counting, most likely by treating
Finding B's multi-bucket eviction as separate from Finding B itself, which `CLAUDE.md` §5 groups
together.

---

## 5. The negative result

**The detector detects at onset. It does not predict. On this data, no density-based detector on
these channels could.**

At a tight 3-hour horizon it finds all four failures with **p = 0.004** — clearly better than chance
— but per-event lead times are **−2, −0, −0, −0 hours** and the mean is **−0.6 h**
(`docs/EVALUATION.md` §2). It fires *as* each failure begins. The large positive leads (+10.1 h at
24 h, +27.2 h at 48 h) appear only when the horizon is widened and **vanish at 3 h**, which means
they were the wider window catching unrelated alarms rather than genuine early warning. The
committed CSV corroborates the mechanism: peak score is higher *after* onset than before it for all
four failures (`docs/figures/scores_around_failures.csv`).

**Mechanism — two independent causes.**

1. **Half the failures have no signature in these channels.** Largest analog effect in the 24 h
   before onset is `Oil_temperature` at **+0.622σ** (failure-2) and **+0.235σ** (failure-3)
   (`docs/figures/effect_sizes.csv`). The information is not there to be found.
2. **A windowed density signal is structurally blind to slow drift.** Once a new regime has filled
   the sliding window it *becomes* the normal and density recovers to baseline, so the elevated-score
   period lasts roughly `window_size` readings after a shift, not for as long as the fault
   (`docs/DATA_NOTES.md` §3). This is correct sliding-window behaviour, not a defect — but it means a
   gradually developing fault is absorbed rather than flagged.

**Methodology, and why it is that way** (`docs/EVALUATION.md` §§1–5):

- **Precision and recall are deliberately not reported.** MetroPT-3 documents exactly **four**
  air-leak failures, in a maintenance report separate from the sensor data (no label column). With
  four positives, one event moves recall by 25% and no defensible confidence interval exists.
  "Papers do report such numbers on this dataset; we think that is a mistake and decline to repeat
  it."
- **What n=4 does support:** per-event detection (four independent yes/no answers), detection lead
  time (per event; the actual value proposition), and false-alarm rate (denominated in *days of
  normal operation*, so independent of the positive count).
- **Alarms are episodes, not samples** — consecutive alerting samples within an hour collapse into
  one episode, which is what an operator sees. Consequence worth knowing: false-alarm rate is
  therefore **not monotone** in the threshold (a low threshold can merge a noisy stretch into one
  long episode while a higher one fragments it). Detection counts *are* monotone.
- **Chance calibration is the control that matters most**, and it changed the reading of the results
  completely. With a 24 h horizon, a detector alarming roughly once a day lands inside a horizon
  most of the time *by construction*. Every operating point is therefore calibrated against 2,000
  random trials with the same alarm budget. The RACE example is the cautionary one: 4/4 detected at
  threshold 1.3, but alerting 17.1% of the time, chance catching 3.21/4, **p = 0.399**.
- **Methods are compared at equal alerting fractions**, chosen as a quantile of each method's own
  score distribution — because RACE's un-windowed counts grow without bound, so comparing at a fixed
  threshold compares nothing.
- **Multiple-comparisons caveat, self-reported:** the horizon table scans 5 horizons × 5 thresholds
  over four events. The low p-values cluster consistently at tight horizons rather than in one lucky
  cell, "which is more reassuring than a single p, but strict Bonferroni over 25 cells (α = 0.002)
  would not pass" (`docs/EVALUATION.md` §2).

---

## 6. Architecture and tech stack

```
MetroPT-3 Parquet ─► Kafka topic ─► Python consumer ─► SW-AKDE sketch (Python or C++ core)
 (replayed)           (KRaft,        (owns logical      + rolling anomaly / quality scoring
                       1 broker)      clock t++)
                                            │
              ┌─────────────────────────────┼──────────────────────────┐
              ▼                             ▼                          ▼
   Prometheus + Grafana              MCP server                  Parquet + PostgreSQL
   + Alertmanager                    (3 agent tools)             (raw stream + alerts)
```

**Scale, stated plainly: single node, single asset, 0.1 Hz.** One Kafka broker, one consumer process,
one air-production unit, a reading every ~10 s (`docs/DATA_NOTES.md` §1). The dataset is 1,516,948
readings — modest. Nothing here is distributed, and `CLAUDE.md` §10 explicitly excludes Spark,
Kubernetes, Terraform and Ray as padding; the "streaming" claim is about pipeline *shape*, not volume.
Phase 2 is not throughput-constrained: at `rows=400` the Python core alone is ~2000× faster than real
time (`docs/PERFORMANCE.md` §"What this means downstream").

| Layer | Choice | Notes |
|---|---|---|
| Sketch core (oracle) | Python 3.12 + numpy | default; the validated reference implementation |
| Sketch core (fast) | C++17 + pybind11 | `src/`; opt-in, bitwise-identical |
| Streaming | Kafka (KRaft) + `confluent-kafka` | single broker |
| Monitoring | Prometheus, Grafana, Alertmanager | thresholds 1.3 / 2.0 from `docs/DATA_NOTES.md` §2 |
| Agent interface | MCP server (Python) | `get_asset_health`, `list_recent_anomalies`, `explain_alert` |
| Storage | Parquet + PostgreSQL | Parquet for the stream, Postgres for alert history |
| Tracking | MLflow | one run per (method, alarm budget) |
| CI | GitHub Actions | 3 jobs; see §7 |

---

## 7. Verification state

**No open items for Phase 5.**

| Claim | State | Evidence |
|---|---|---|
| Compiles under MSVC (local) | Verified | `MSVC 1944`, `/O2`, zero warnings at `/W3` |
| Compiles under gcc | Verified | CI run 30979739606 + 30981570447, ubuntu-latest: `native core built: gcc 13.3.0` |
| Compiles under a second MSVC | Verified | same runs, windows-latest: `native core built: MSVC 1951` |
| `pip install -e .` builds it | Verified | both native-core jobs, "Install with the native core" → success (no custom build driver needed on a clean runner) |
| Pure-Python install works | Verified | `python-core` job, `SWAKDE_SKIP_NATIVE=1`, "Confirm the native core really is absent" → success |
| Parity green locally | Verified | 130 pass with the extension, 93 + 37 skipped without |
| **CI enforces parity** | **Verified** | run **30981570447** (`35f66ba`): 37 parity tests **executed** (0 skipped) in both native jobs; 37 skipped in the pure-Python control |
| **Native core loads in the container** | **Verified** | see below |
| CI overall | **Green** | runs 30981570447 (`35f66ba`) and 30981820742 (`106d5e8`), all three jobs |

**Parity is enforced on gcc 13.3.0 and MSVC 1951** (the two CI native jobs). A third compiler,
**gcc 14.2.0**, builds the container image — but note precisely: the parity *suite* was not executed
inside the container, so gcc 14.2.0 is verified to **build and load** the core, not to have passed
parity.

**Container verification** — `docker compose build consumer && docker compose up -d consumer`, then
`docker compose logs consumer | grep "Sketch core"` printed exactly:

```
swakde-consumer  | Sketch core: native
```

From inside the running container: `AVAILABLE=True`, `compiler = gcc 14.2.0`, module
`/app/sketch/_native.cpython-312-x86_64-linux-gnu.so`, resolved backend `native`, store type
`sketch._native.CellArray` (the C++ class, not `PythonCellStore`), `SKETCH_BACKEND` **unset** (so
`auto` resolved to native on its own — the exact path that could have silently fallen back), and no
compiler on `PATH` (the purge worked; `libstdc++6` survived `autoremove`). *This is live-run
evidence, reproducible by the commands above, not a committed output file.*

**Parity is asserted as bitwise equality, not tolerance** — `tests/test_native_parity.py` asserts
`==` on query output for both kernels and compares per-cell state down to bucket timestamps and
sizes; there is no `pytest.approx` in the file. Both cores sum exactly representable doubles in the
same order, so there is no floating-point reordering to excuse a difference.

---

## 8. What is NOT claimed

- **Not novel research.** The SW-AKDE algorithm is Danait, Das & Bhore's. `CLAUDE.md` §2 is explicit
  that re-implementing the paper is no novelty, and no document here claims a new algorithm.
- **The paper's published results are not challenged.** Findings E, G and H are inert at the `k=1`
  the paper uses for all experiments. Findings A and F *are* in Algorithm 2 and active at `k=1`, but
  we have **not** attempted to quantify what they would change in the paper's reported figures, and
  therefore assert nothing about them (`README.md` §Correctness).
- **Not a working predictive detector for MetroPT.** It detects at onset with ~0 lead time; two of
  the four failures have no analog signature beforehand (§5).
- **The sketch is the wrong tool for this workload.** At `dim=7` it costs 26–236× more memory than
  storing the window on a near-trough basis, and ~385× at `rows=400` on a peak basis (§9.1);
  `docs/EVALUATION.md` §4 recommends exact windowed KDE over our own sketch here. The applicability
  finding, not the detector, is the contribution.
- **The peak memory figure is not a proven maximum.** The 77.6 MB / ~385× peak comes from one sample
  3,000 ticks into one compaction cycle. `memory_check` does not track a running maximum, so a true
  peak-footprint number would need that added (§9.1).
- **Not distributed, not big data.** Single node, single asset, 0.1 Hz, 1.5M readings (§6).
- **The speedup is against our own optimized Python core**, not naive Python. The 2.9× from Phase 1.5
  is *not* folded in (`docs/PERFORMANCE.md` line 119).
- **Timed figures are noisy.** Per-row speedup ratios move up to ~3× between runs on the development
  desktop; only the structural memory figures reproduce exactly (§3.1).
- **Phase 6 (adaptive window size) was not attempted.**
- **`make` was never run** on the development machine; targets were exercised as module invocations.

---

## 9. Internal disagreements — investigated and resolved

All four substantive disagreements found while writing this document have now been traced to root
cause and the source docs corrected. Recorded here because the *reason* each arose is more useful
than the fix.

### 9.1 RESOLVED — the 761 vs 787 bytes-per-cell split was a compaction-cycle phase artifact

This was the important one: the `dim > ~2.3 × rows` correction is derived from the 345/761 pair, so
if 761 were wrong the correction would be too.

**Both numbers are correct.** Both were re-run and reproduce exactly. The two code paths feed
different numbers of readings and therefore stop at different points in the compaction cycle:

- `scripts/memory_check.py --rows-sweep` slices `.iloc[:records]` **then** discards ~2,000 warmup
  readings → feeds ~**148,000**, stopping **400 ticks** past a sweep.
- `scripts/benchmark_native.py --mode memory` slices `records + warmup` → feeds exactly
  **150,000**, stopping **2,400 ticks** past a sweep.

Compaction sweeps only once per `window_size` = 3,600 ticks, so dead cells accumulate between
sweeps and the live count **sawtooths by roughly 2×**. Measured directly at `rows=400`:

| clock | ticks since sweep | live cells | cell MB | B/cell |
|---:|---:|---:|---:|---:|
| 147,000 | 3,000 | 116,989 | 77.6 | 663 |
| 147,600 | 0 (just swept) | 58,421 | 46.8 | 802 |
| 148,000 | 400 | 60,291 | 47.5 | **788** ← matches memory_check (60,330 / 47.5 / 787) |
| 150,000 | 2,400 | 83,920 | 63.9 | **761** ← matches benchmark_native exactly |

Forcing a sweep at t=150,000 drops 17,594 cells (83,920 → 66,326) and moves bytes-per-cell from
**761 to 788** — i.e. onto memory_check's figure. That is the mechanism, confirmed.

**Which basis is correct for the crossover derivation: the ratio, and it is sound.** The 761 and the
345 come from the same run at the same clock with both cores holding an identical 83,920 cells, so
they are phase-matched and the 345/761 factor is valid. What is *not* a stable constant is the
absolute bytes-per-cell figure. Two consequences, both now written into `docs/EVALUATION.md`:

1. `bytes_per_cell` "is constant ~590–800 bytes" is really the span of the sawtooth, not a constant.
   The observed 663–802 range sits inside the doc's quoted range, which is why this went unnoticed.
2. **The absolute megabytes in the boundary table are near-trough samples and understate the
   footprint you must provision by ~1.6×** (77.6 vs 47.5 MB at `rows=400`). On a peak basis the
   `rows=400` sketch is closer to ~385× exact storage than the 236× quoted. **This makes the negative
   result stronger, not weaker.**

The **530** B/cell figure in `docs/PERFORMANCE.md` §"Cell reclamation" is a third phase sample at a
different record count (400,000) and is separately explained there. Caveat that remains: the 77.6 MB
peak is a single observed sample 3,000 ticks into one cycle, not a proven maximum over the run — a
true peak measurement would need `memory_check` to track a running maximum, which it does not.

### 9.2 RESOLVED — crossover at `rows=50` is 179, and the CSV's 180.56 is the artifact

The opposite of the initial reading. `scripts/make_plots.py` line 188 **hardcodes**
`measured = {50: 5.2, 400: 47.5}` — MB values transcribed by hand from `memory_check` output — and
line 229 then recomputes `crossover_dim = mb * 1e6 / (window * 8)` from those *already-rounded*
inputs, giving 5.2e6/28800 = **180.56**. `memory_check` computes the same quantity from the exact
byte count and prints **179**. So 179 is the more accurate figure and the doc was right; the CSV
carries a rounding artifact. Kept at 179.

**Separate fragility this exposed, not fixed:** the figure and its CSV are generated from
hand-transcribed constants, so they can silently drift from the measurement they claim to plot.
Worth wiring `make_plots` to the measurement output rather than a literal.

### 9.3 RESOLVED — "~500 dimensions at 100 rows" corrected to ~420

`docs/EVALUATION.md` contradicted its own table two sections earlier (418). Corrected to ~420, with
the previous wording noted inline so the change is visible.

### 9.4 RESOLVED — the defect count is seven, not eight

Corrected in `README.md` and `docs/EVALUATION.md`; full enumeration in §4 above, with README's own
breakdown extended to 2 + 3 + 2 so its arithmetic closes.

### 9.5 Not a disagreement

`CLAUDE.md` §11 states MetroPT is sampled "every 10s (0.1 Hz — measured, despite the dataset docs
saying 1 Hz)", which agrees with `docs/DATA_NOTES.md` §1. Recorded only because the original brief
said 1 Hz and the correction propagated; no live conflict remains.

---

## 10. Reproducing everything

```bash
# setup
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,streaming]"
.venv/Scripts/python.exe -m pytest                       # 130 with native core, 93+37 without

# native core (optional; Python core is the default and the oracle)
.venv/Scripts/python.exe -m pip install "pybind11>=2.13" "setuptools>=68"
.venv/Scripts/python.exe -m scripts.build_native
.venv/Scripts/python.exe -m scripts.benchmark_native                              # §3.1 tables
.venv/Scripts/python.exe -m scripts.benchmark_native --tail-check                 # p99 attribution
.venv/Scripts/python.exe -m scripts.benchmark_native --eval-parity --records 30000

# evaluation (needs the dataset: python -m scripts.download_data, ~208 MB, not committed)
.venv/Scripts/python.exe -m scripts.run_evaluation --method swakde   # then race, exact
.venv/Scripts/python.exe -m scripts.compare_methods
.venv/Scripts/python.exe -m scripts.horizon_sensitivity
.venv/Scripts/python.exe -m scripts.diagnose_failures
.venv/Scripts/python.exe -m scripts.memory_check --records 150000 --rows-sweep    # §3.4
.venv/Scripts/python.exe -m scripts.make_plots                                    # figures + CSVs

# container native-core check (§7)
docker compose build consumer && docker compose up -d consumer
docker compose logs consumer | grep "Sketch core"        # must print: Sketch core: native
```

`README.md` §"Without `make`" carries the full target ↔ command equivalence table.

---

## 11. Source index

| Document | What it is authoritative for |
|---|---|
| `CLAUDE.md` | project brief; novelty framing (§2); findings summary (§6); phase DoDs (§12); stack rationale (§10) |
| `docs/REFERENCE_NOTES.md` | line-level reference audit; Findings B, E, F, G, H evidence tables |
| `docs/PERFORMANCE.md` | Phase 1.5 profiling; throughput; cell reclamation; all Phase 5 numbers; verification-state table |
| `docs/EVALUATION.md` | evaluation methodology; detection results; the applicability boundary |
| `docs/DATA_NOTES.md` | what the real data forced: cadence, detector tuning, transience |
| `docs/figures/*.csv` | committed exact values behind every evaluation figure |
| `README.md` | status, correctness summary, make-free command equivalents |
| CI runs 30979739606 / 30981570447 / 30981820742 | compiler evidence, test counts, parity execution |
