# CLAUDE.md — Real-Time Streaming Anomaly & Data-Quality Monitor

> Definitive project brief for Claude Code. Read fully before writing any code. This supersedes any earlier version. It contains: what the source algorithm is, what its authors did, what *we* are contributing, the architecture, the stack, and — critically — everything we learned from auditing the authors' reference code, including real bugs you must NOT copy and the exact strategy for validating our own re-implementation.
>
> Companion file: `docs/REFERENCE_NOTES.md` (the full line-level audit). This brief distills the actionable parts; go to that file for exact file/line pointers.

---

## 0. TL;DR

We are turning a 2025 research algorithm — a **sliding-window Approximate Kernel Density Estimation (SW-AKDE) sketch** — into a **production-shaped, real-time anomaly & data-quality monitoring system** over an **industrial equipment sensor stream**, and exposing it to AI agents through an **MCP server**.

- The algorithm already exists as an **unoptimized, buggy Python research prototype**. Our contribution is the **engineering** (clean, optimized, genuinely streaming, deployed) and the **application** (industrial-sensor anomaly detection) — NOT a new algorithm.
- We **re-implement the sketch cleanly ourselves** (the reference has a real correctness bug and bad hygiene), **reuse the reference's clean utility primitives**, and **validate against sketch-independent brute-force ground truth** — never against the reference's buggy sketch outputs.
- Do not, in any doc/commit, claim we invented a new algorithm. We productionize and apply an existing one.

---

## 1. Source paper — what the authors did

- **Paper:** *Sublinear Sketches for Approximate Nearest Neighbor and Kernel Density Estimation*, Danait, Das, Bhore (IIT Bombay), arXiv:2510.23039 (Oct 2025). https://arxiv.org/abs/2510.23039
- **Reference code:** https://github.com/VedTheKnight/Streaming-ANN-and-Sliding-Window-KDE

The paper has two halves. **We use only the KDE half. Ignore the ANN half entirely.**

**What the SW-AKDE contribution is:** a compact, fixed-memory sketch that answers "how dense is the region around this query point, considering only the last N stream elements?" It is built by composing two known pieces:
- **RACE** (Coleman & Shrivastava, 2020) — an array of LSH-bucketed counters that estimates kernel density over a stream. `L` rows; each row hashes a point with `k` concatenated locality-sensitive hash (LSH) functions into a cell; the cell counts collisions; density ≈ average cell count across rows.
- **Exponential Histograms / DGIM** (Datar–Gionis–Indyk–Motwani, 2002) — a sliding-window counter that estimates "how many of the last N items landed here" in `O(log N · 1/eps)` space instead of `O(N)`.

The novelty they claim: replacing each RACE integer counter with an Exponential Histogram, giving RACE **sliding-window semantics** (old data expires). They state this is, to their knowledge, the **first sublinear sketch guarantee for A-KDE in the sliding-window model**. That research novelty is **theirs, not ours.**

**Why density → anomaly detection (our use):** if the density of *recent* sensor patterns around a normal operating point suddenly drops (or a new region lights up), that's an emerging fault or data-quality break. The RACE counter primitive was in fact originally invented for anomaly detection, so this application is natural.

---

## 2. Our novelty (keep all docs honest)

Re-implementing the paper as-is = **no novelty** (their code exists). Our value is entirely in what they did NOT do:

1. **Lever 1 — Engineering (CORE).** No optimized, genuinely-streaming, throughput-benchmarked, or deployed implementation exists. We build a clean, tested one (Python reference port first, then an optimized C++ core).
2. **Lever 2 — Application (CORE).** They never applied this to industrial-sensor anomaly detection. We do (MetroPT), evaluated against ground-truth failures. **Outcome (measured, Phase 3):** the sketch matches exact windowed KDE's detection quality to within noise, and detects all four documented failures at onset (p=0.004) — but at MetroPT's seven channels it costs **26–236x more memory than simply storing the window**, because the crossover is around `dim > 5 x rows`. The application therefore sits outside the regime where the algorithm's central advantage exists. We report that boundary as the contribution rather than working around it; see `docs/EVALUATION.md`. Do not describe Lever 2 as a straightforward success.
3. **Lever 3 — Research extension (OPTIONAL STRETCH).** The paper's future-work flags **adaptive sliding-window-size selection** from the data distribution. Attempt only after Levers 1–2 are solid.

MVP scope = Levers 1–2. Lever 3 is a stretch for depth / publishability.

---

## 3. Two audiences this must satisfy (why the design has two faces)

Selection for a university-run industry project; we want to qualify for either:
- **Meta (primary)** — "Scalable Real-Time Data Engineering": production streaming pipeline + live data-quality/anomaly monitoring + dashboards + alerting.
- **Siemens (fallback)** — "AI-Powered Predictive Asset Management": an MCP server letting AI agents query and act on industrial equipment data.

---

## 4. The reference repo — critical facts

**Two branches, NOT the same. The code we need is ONLY on `racecms-benchmark`.**
- `main` (default): **incomplete** — only `Code/StreamingANN/` (the ANN half we don't use). The entire `Code/SlidingWindowKDE/` directory is **missing** from `main`.
- `racecms-benchmark`: **complete** — has `Code/SlidingWindowKDE/`.

We audited it at commit **`cc7d7bc1f3902939071a7430a9187be8f05cb237`** — pin to this. The repo is loosely maintained on a non-default branch and could be force-pushed, so keep an **archived copy** of that commit; do not rely on it staying live.

**The reference is read-only material.** We build in a **separate clean repo**. We do NOT build inside their tree, and we do NOT fork-and-patch it (bad hygiene, see §7).

---

## 5. What the reference code actually contains (audit summary)

All paths under `Code/SlidingWindowKDE/`. Full line-level detail in `docs/REFERENCE_NOTES.md`.

**The utility layer — clean, correct, directly reusable (port these, with tests; no need to reinvent):**
- `Exponential_Histogram.py` — DGIM sliding-window counter (`ExpHst`). Update path `new_bucket(t)`; query path `count_est()` returns `total - last/2` (textbook DGIM). **Caveat: single-step expiry — see Finding B in §6.**
- `buckets_DS.py` — trivial bucket POD (size + timestamp).
- `angular_hash.py` — one-bit SimHash (random-hyperplane) LSH for cosine similarity; `k` bits concatenated → `2^k` cells.
- `p_stable.py` — `PStableLSH` (Datar et al. Euclidean LSH) + `UniversalHasher` (Carter–Wegman fold to a bounded cell id). Used only by the L2 sketch.

**The sketch layer — right architecture, but carries a real bug (RE-IMPLEMENT, do not copy):**
- `Ang_hash_AKDE.py` (`RACE_Ah`) — SW-AKDE with the angular kernel. `L` rows; each cell holds an `ExpHst` instead of an int. `update_counter(data, t)` = update; `query1(data)` = **mean over rows** (this is the one to use). `query2` (median-of-means) exists but is **unused and vestigial** — see Finding C. `hash_range` constructor arg is **dead** here (real range is `2^k`).
- `L2_hash_AKDE.py` (`RACE_L2`) — same scheme, Euclidean kernel + `UniversalHasher`. `hash_range`/`R` is **live** here. Imports `torch` but it is **100% dead** (one unused `device` line) — drop it. Also defines `l2_lsh_collision_probability`, a ground-truth helper used by drivers.

**The baseline — clean, no windowing bug (USE as a baseline/oracle, don't rebuild):**
- `RACE_19.py` (`RACE_1`) — plain un-windowed RACE (integer counters, no eviction). Its first-hit handling is **correct** (unlike the two AKDE files). Also provides `angle_between_vectors` used for brute-force ground truth.

**Baseline #2 (bonus):**
- `TAKDE/takde_alg.py` — a competing batch-adaptive sliding-window KDE method. Algorithm is complete/self-contained (pure NumPy) but `main.py`/`generate_data.py` have **hardcoded non-portable paths** — trivial to fix, but not runnable as-is. Good as a second evaluation baseline.

**Reusable-vs-rewrite verdict:**

| File | Verdict |
|---|---|
| `Exponential_Histogram.py` | Reusable **with fixes** (Finding B multi-bucket eviction; Finding F expire-on-read; window-boundary off-by-one) |
| `buckets_DS.py` | Reusable |
| `angular_hash.py` | Reusable |
| `p_stable.py` | Reusable |
| `Ang_hash_AKDE.py` | Reusable **with fixes** (Findings A, E, F; drop dead `hash_range`) |
| `L2_hash_AKDE.py` | Reusable **with fixes** (Findings A, F — check E for its own code path; drop dead `torch`) |
| `RACE_19.py` | Baseline **with fix** (Finding E — no first-arrival bug, but same cell-code bug) |

---

## 6. The findings that shape OUR build (build these in from day one)

> A–D came from the original audit. **E and F were found while writing the Phase-1 port** (2026-08-02) and are at least as important; both are in `docs/REFERENCE_NOTES.md` with evidence. Two smaller EH fixes (multi-bucket eviction, window-boundary off-by-one) are documented there too.

**Finding A — First-arrival counting bug (in both `*_AKDE.py`, NOT in `RACE_19.py`).**
On a cell's very first hit, the code creates an empty `ExpHst` but **never records that first element** (`.new_bucket()` isn't called on the create branch). So every cell silently drops its first arrival. Impact is concentrated in the **low-density / sparse-cell regime** (recently-discovered cells, singletons — i.e. large `L`/`k`), and is ~zero for cells hit early-and-often (that first hit would expire from the window anyway). **Action:** fix it in our port (record the first element). **Consequence:** do NOT validate our port by matching their sketch outputs — their numbers are slightly biased low. **Confirmed 2026-08-02:** this bug is baked into the paper's own Algorithm 2 pseudocode (§4.1), not just the reference implementation — the "if empty: create EH" branch has no corresponding "add 1," only the else branch does. There is no "correct" version of this step anywhere in the source material; our fix is a deliberate deviation from both.

**Finding B — Single-step expiry assumption (logical clock required).**
Their `ExpHst` checks only the *one* oldest bucket per update, which is correct **only if `t` increases by exactly 1 per element**. Every driver feeds sequential `t = j+1`, so it's fine for them. But **MetroPT is real 1 Hz wall-clock data** — timestamps can repeat, skip, or batch. **Action:** in our pipeline, `t` MUST be a **monotonic per-event logical counter (increment by 1 per reading)**, NOT the sensor wall-clock timestamp — OR convert EH expiry to a `while` loop that evicts all expired buckets. Pick the logical-counter approach for the MVP; document it. This is a silent-failure trap if ignored.

**Finding C — Use the mean estimator (matches the paper).**
Drivers use `query1` (plain mean). The paper **specifies the mean** for SW-AKDE ("average of ACE estimates over L repetitions"); median-of-means is for plain RACE. So mean-vs-median is NOT a code/paper divergence — the unused `query2` is vestigial. **Action:** our port uses the **mean** for SW-AKDE. **Confirmed 2026-08-02** by reading the paper's §4.1 directly: "For SW-AKDE, we will take the average of ACE estimates over L independent repetitions."

**Finding E — Cell code only encodes Hamming weight (in `Ang_hash_AKDE.py` AND `RACE_19.py`).**
Both build the `k`-bit cell code as `r = bit*2 + r` instead of `r = r*2 + bit`, which collapses to `r = 2 * (count of set bits)` — *which* hashes fired is lost, only *how many*. Correct at `k=1`, badly wrong beyond: measured collision probability at `k=8` is **0.242 vs 0.039 theoretical (~6x)**. This nullifies the LSH amplification Theorem 2.3/2.4 depends on. **Action:** use standard shift-and-append (`code = code*2 + bit`) in both our sketch and our RACE baseline.

**Finding G — Euclidean cell code sums the `k` hashes (the L2 analogue of E, and worse).**
`L2_hash_AKDE.py` keys cells on `s = sum(r)`, which is order-invariant and maps `R^k` tuples onto `≤ k*R` sums that concentrate near the mean. Measured on 4,000 points at `k=5`, `R=2^20`: **101 distinct cells vs 3,384** with a correct fold (97.5% vs 15.4% collision rate). **Action:** fold with a polynomial hash (`code = (code*MULT + value) mod R`).

**Finding H — the L2 brute-force ground truth omits `** k`.**
`compute_true_kde_l2` sums the single-hash collision probability without raising it to `k`, while the angular helper in the same file does. **Action:** apply `** k` in our `compute_true_kde_l2`.

> **Honesty caveat on E, G and H — they do NOT invalidate the paper's published results.** The paper sets the concatenation parameter to 1 for all experiments (§5.2), and every documented reference invocation passes `--b 1`. All three bugs are inert at `k=1`. They are *latent*: they break as soon as `k>1`, which is the regime the whole LSH-amplification argument is about. State it exactly that way — do not claim their experiments are wrong.

**Finding F — Cold cells never expire; queries return frozen counts.**
`ExpHst` expires only inside its update path, and `count_est()` takes no time argument — so a cell that stops being hit reports a stale count forever. The paper's Algorithm 2 query procedure has the same shape, so this is faithful-but-broken. **This is the finding that matters most for us:** a region going quiet *is* our anomaly signal, and without expiry-on-read that density never decays. **Action:** `count_estimate(t)` takes the current logical clock and expires before reading; `query(x, t)` threads it through. Also bounds memory for cold cells.

**Finding D — Validate against brute-force ground truth, not the sketches.**
The trustworthy oracle is the **sketch-independent brute-force KDE** the reference already contains: `angle_between_vectors`-based windowed KDE and the vectorized `compute_true_kde_angular` / `compute_true_kde_l2`. These compute the *true* windowed density directly, with no LSH/sketch involved. **Action:** validate our port against these (details in §9), NOT against `*_AKDE` outputs.

---

## 7. Reference hygiene issues (why we don't build on their tree)

- ~86% of tracked files in that dir are committed artifacts (`__pycache__`, `.pyc`, `.log`, `.npy`, `.npz`), including bytecode for **CPython 3.8, 3.10, and 3.13 simultaneously** — contradicting the README's "Python 3.12.0." **Committed numeric outputs are NOT authoritative** — regenerate anything you need from a clean env.
- `simulate_AH.py` always writes to `Synthetic_data_outputs_L2/` regardless of `--lsh` (contradicts README, breaks the Angular plot inputs).
- Dead/commented plotting: `sketch_size.py`'s plot call is commented out; `mc_compare.py`'s plot block is inside a triple-quoted string (never runs). Use `mc_plot_compare.py` / `MC_plot_sketchsize.py` instead.
- `compare.py` is a stale duplicate of `compare2.py` (undocumented).
- Hardcoded paths in `TAKDE/main.py` and `TAKDE/generate_data.py`.
- No `requirements.txt` / `setup.py` / `pyproject.toml` anywhere; deps unpinned. Practical minimum: `numpy`, `matplotlib`, `scipy` (L2 only), `tqdm` (only for `mc_compare.py`). **`torch` is not needed.**

---

## 8. Architecture (what we build)

```
 MetroPT sensor data ─► Kafka topic ─► Stream consumer ─►  SW-AKDE sketch (OUR clean port)
 (replayed as a live     (replay)      (logical clock t++,   + rolling anomaly/quality scoring
  ~0.1 Hz stream)                        per-reading update
                                         + density query)
                                                 │
                    ┌────────────────────────────┼──────────────────────────┐
                    ▼                             ▼                           ▼
            Prometheus + Grafana            MCP server                 Parquet + PostgreSQL
            + Alertmanager                  (agent-queryable tools)    (raw stream + alert history)
            ── the "Meta face" ──           ── the "Siemens face" ──
```

MCP tools (minimum): `get_asset_health`, `list_recent_anomalies`, `explain_alert`.

---

## 9. Validation strategy (three tiers — the correctness backbone)

Never validate against the reference sketch outputs (Finding A). Validate bottom-up against sketch-independent truth:

1. **EH correctness (unit).** Feed a known 0/1 sequence into our Exponential Histogram; assert `count_est()` matches a brute-force count of the last N. **Fix and prove the first-arrival case here** with an explicit test.
2. **Un-windowed sketch.** Set window ≥ stream length (nothing expires); compare our sketch against (a) `RACE_19.py` and (b) brute-force all-stream KDE. They should agree — RACE is un-bugged.
3. **Windowed sketch.** Compare our sketch against **brute-force last-N KDE** (`angle_between_vectors` / `compute_true_kde_*`), across the parameter sweeps the paper uses (rows in {100…3200}, window in {64…2048}). This reproduces the paper's accuracy story with a *correct* implementation.

CI runs tiers 1–2 on every push; tier 3 is the evaluation notebook.

---

## 10. Tech stack (and what NOT to add)

Rule: every dependency needs a specific, project-grounded reason. No padding.

| Layer | Tool | Why |
|---|---|---|
| Sketch core (reference port) | **Python 3.12** | Clean, correct, tested port validated per §9. First deliverable. |
| Sketch core (optimized) | **C++17 + pybind11** | Lever-1 engineering novelty: optimized native hot path (EH update + query), same Python interface. **Only after the Python port passes §9.** |
| Streaming | **Kafka** | Replay sensor data as a real live stream. |
| Stream processing | **Python consumer** (`confluent-kafka`) | Lean; sufficient for MetroPT's ~0.1 Hz single-asset feed. Owns the **logical clock** (Finding B). |
| Monitoring | **Prometheus + Grafana + Alertmanager** | The Meta quality-monitoring + alerting face. |
| Agent interface | **MCP server (Python)** | The Siemens face. |
| Storage | **Parquet + PostgreSQL** | Parquet for raw/replayed data; Postgres for alerts + metadata. |
| Experiment tracking | **MLflow** | Parameter sweeps vs accuracy/memory; reproduce the paper's plots (correctly). |
| Infra / CI | **Docker + GitHub Actions** | `docker compose` local stack; CI runs tests + Python-vs-C++ parity. |
| Tracing (optional) | **OpenTelemetry** | Only for end-to-end observability depth; else skip. |
| Analysis | **Jupyter + pandas + matplotlib** | Benchmark plots + writeup. |

**Excluded as padding — do NOT add (stop and flag if a task seems to need one):** Ray/RLlib, Kubernetes/EKS/Helm, Terraform, multi-service AWS, and **Apache Spark** (only justified if we deliberately go **multi-asset**; single-asset MetroPT does not need it).

---

## 11. Datasets

- **Primary — MetroPT.** Real metro-train Air Production Unit sensors (pressure, temperature, motor current), continuous flow sampled every 10s (0.1 Hz -- measured, despite the dataset docs saying 1 Hz), **ground-truth anomalies** from maintenance reports. Paper/DOI: https://www.nature.com/articles/s41597-022-01877-3. Provide a `scripts/download_data.py`; do NOT commit the data. Note: modest in volume — the "big data" claim is about the **streaming architecture**, not raw size. Keep that honest.
- **Optional — NASA C-MAPSS.** Run-to-failure turbofan time series (NASA PCoE). For later cross-asset generalization only.

---

## 12. Build plan (ordered; each phase has a definition of done)

**Phase 0 — Setup.** New clean repo. Copy in `docs/REFERENCE_NOTES.md`. Archived reference tarball kept OUTSIDE this repo. **DoD:** clean repo, this `CLAUDE.md` at root, notes in `docs/`.

**Phase 1 — Validated pure-Python port.** Re-implement the utility layer + SW-AKDE sketch as an importable `sketch/` package with tests. **Fix Finding A, apply Findings B & C.** Pass all three validation tiers (§9). **DoD:** our windowed KDE matches brute-force truth within tolerance; `pytest` green; parity with `RACE_19` in the un-windowed limit. **Hard gate before any optimization.**

**Phase 2 — Streaming skeleton (Meta face).** Kafka via `docker compose`; producer replays MetroPT with a **logical clock**; consumer updates the sketch + rolling anomaly/quality scores; Prometheus metrics; one Grafana dashboard + one Alertmanager rule. **DoD:** `docker compose up` → live dashboard reacting to the stream, alert fires on an injected anomaly.

**Phase 3 — Anomaly detection + evaluation (Lever 2).** Turn density into a labeled detector; evaluate vs MetroPT ground-truth failures (precision/recall, detection lead time); add TAKDE + RACE baselines. **DoD:** evaluation report/notebook with metrics + plots vs baselines, logged to MLflow.

**Phase 4 — MCP server (Siemens face).** Minimal server exposing `get_asset_health`, `list_recent_anomalies`, `explain_alert`, backed by the live sketch + Postgres. **DoD:** an MCP client/agent queries health + recent anomalies end-to-end.

**Phase 5 — Optimized native core (Lever 1).** C++17 + pybind11 hot path behind the same interface; honest benchmark vs the Python port (throughput, latency, memory); **accuracy must still match Phase 1.** **DoD:** benchmark report showing speedup + identical accuracy.

**Phase 6 (optional stretch) — Adaptive window size (Lever 3).** Only if time allows.

**MVP for the deadline = Phases 0–4.**

---

## 13. Conventions for Claude Code

- **Correctness before speed.** Never optimize a component not yet validated per §9.
- **Keep both cores.** The Python port stays as the oracle after the C++ core exists; CI compares them.
- **Clean repo.** No committed caches, datasets, or result binaries — `.gitignore` them; provide download/generation scripts.
- **Reproducible.** Everything via `docker compose` / `Makefile`; pin versions.
- **Honest docs & commits.** Reflect the novelty framing (§2); cite the paper; record the reference commit hash. Never claim a new algorithm.
- **Don't copy the bug.** Re-implement the sketch; apply Findings A–D. Do not lift `*_AKDE.py` verbatim.
- **Flag scope creep.** If a task seems to need an excluded tool (§10), stop and ask.

---

## 14. References

1. Danait, Das, Bhore. *Sublinear Sketches for ANN and KDE.* arXiv:2510.23039, 2025. https://arxiv.org/abs/2510.23039
2. Reference code (branch `racecms-benchmark`, commit `cc7d7bc1f…`): https://github.com/VedTheKnight/Streaming-ANN-and-Sliding-Window-KDE
3. Veloso et al. *MetroPT dataset.* Scientific Data, 2022. https://www.nature.com/articles/s41597-022-01877-3
4. Coleman & Shrivastava. *Sub-linear RACE sketches for A-KDE on streaming data.* WWW 2020.
5. Datar, Gionis, Indyk, Motwani. *Maintaining stream statistics over sliding windows.* SIAM J. Comput., 2002.
6. Wang, Ding, Shahrampour. *TAKDE: Temporal Adaptive KDE.* IEEE TPAMI, 2023.

---

*Scope note: Levers 1–2 are the committed core; Lever 3 is an explicit optional stretch. Full line-level reference audit lives in `docs/REFERENCE_NOTES.md`.*
