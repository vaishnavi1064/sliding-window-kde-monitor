# Real-Time Streaming Anomaly & Data-Quality Monitor

A production-shaped streaming anomaly and data-quality monitoring system built on a
**sliding-window Approximate Kernel Density Estimation (SW-AKDE) sketch**, applied to
industrial equipment sensor data and exposed to AI agents through an MCP server.

## What this is (and is not)

The SW-AKDE algorithm is **not ours**. It comes from Danait, Das & Bhore,
*Sublinear Sketches for Approximate Nearest Neighbor and Kernel Density Estimation*
([arXiv:2510.23039](https://arxiv.org/abs/2510.23039), Oct 2025), which composes two known
pieces — RACE (Coleman & Shrivastava, WWW 2020) and Exponential Histograms (Datar, Gionis,
Indyk & Motwani, 2002) — to give RACE sliding-window semantics. That research contribution is
theirs.

**Our contribution is the engineering and the application:**

1. **Engineering** — the authors' reference implementation is an unoptimized research
   prototype. We re-implement the sketch cleanly, with tests, validated against
   sketch-independent ground truth, and (in progress) an optimized native core.
2. **Application** — the paper never applies this to industrial-sensor anomaly detection.
   We do, against a real dataset with documented ground-truth failures.

We do not claim a new algorithm.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Repo, reference audit | Done |
| 1 | Validated pure-Python sketch (angular kernel) | Done |
| 1.5 | Euclidean kernel; profiling + 2.9x vectorization | Done |
| 2 | Kafka streaming, Prometheus/Grafana/Alertmanager | Done — alert verified firing end to end |
| 3 | Anomaly detector + MetroPT evaluation | Done |
| 4 | MCP server | Done — three tools verified end to end |
| 5 | C++17 + pybind11 optimized core | Planned |
| 6 | Adaptive window size (research extension) | Stretch |

88 tests green. Engineering log in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md); findings from
running against the real data in [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md); evaluation
methodology in [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Headline result

Full details in [`docs/EVALUATION.md`](docs/EVALUATION.md). Three things, and the
third is the one that matters.

**The sketch works, and matches exact windowed KDE to within noise.** At equal alarm
budgets, SW-AKDE and brute-force exact KDE agree at every operating point (4/4 failures
at p = 0.050 versus p = 0.049). Un-windowed RACE is genuinely worse (3/4 at the same
budget), so sliding-window semantics do buy something real.

**It detects at onset, but does not predict.** All four documented failures are found at
a 3-hour horizon with p = 0.004 — clearly better than chance — but with lead times of
about zero. The large positive leads that appear at wider horizons vanish at 3h, meaning
they were the wider window catching unrelated alarms. Two of the four failures have
almost no signature in these sensors beforehand (+0.62σ and +0.24σ), so no
density-based detector could predict them.

**At seven channels, the sketch costs 26–236× more memory than simply storing the
window** — and this is the contribution. Its footprint is independent of dimension while
exact storage grows with it, so there is a crossover, measured here at roughly

> `dim > 5 × rows`

Exact windowed KDE needs 0.20 MB for MetroPT; the sketch needs 5–48 MB for the same
detection quality. Because accuracy demands rows and memory is linear in rows, **the
dimension required to justify the sketch grows with the accuracy you want** — a tension
the source paper does not discuss. Its own experiments used 103-, 200- and
384-dimensional data, comfortably inside the useful regime; a seven-channel sensor feed
is two orders of magnitude outside it.

So for MetroPT specifically we would recommend exact windowed KDE over our own sketch.
We report that boundary rather than engineering around it: knowing where the line sits,
measured, is more useful than a demonstration that avoided the question.

## Correctness

We validate bottom-up against ground truth that involves no sketch at all, never against the
reference implementation's outputs — because auditing that implementation turned up eight real
bugs, documented with evidence in [`docs/REFERENCE_NOTES.md`](docs/REFERENCE_NOTES.md). Some
trace back to the paper's own pseudocode rather than just the code:

**Two are in the published algorithm itself**, not only in the reference code — both in
Algorithm 2 of arXiv:2510.23039 (§4.1), and both active at the `p = 1` setting the paper
uses for all its experiments:

- **Finding A — every cell silently drops its first arrival.** Algorithm 2's
  preprocessing loop reads `if A[i,j] is empty then Create an Exponential Histogram …
  else Add a 1 …`: the create branch has no corresponding "Add a 1", so the element that
  created the cell is never counted. The reference repeats it
  (`Ang_hash_AKDE.py` lines 25-28).
  Test: [`test_every_cells_first_arrival_is_counted`](tests/test_sw_akde_windowed.py) —
  one element gives density 1.0 correctly and exactly 0.0 under the published branch
  structure. The tier-2 convergence test cannot see this (a one-per-cell undercount
  measures 0.074 vs 0.063 mean relative error, both inside its 0.30 tolerance), which is
  why it is tested separately.
- **Finding F — cells that stop receiving data never expire, so their density is frozen
  forever.** Algorithm 2's query procedure reads
  `c ← estimate of count in the Exponential Histogram at A[i, h_i(q)]` — no time
  argument, so it cannot expire before reading; the reference's `count_est()` likewise
  takes no timestamp. This is the one that matters most here: a region going quiet *is*
  the anomaly signal, and without expiry-on-read that density never decays.
  Test: [`test_window_actually_expires_old_data`](tests/test_sw_akde_windowed.py) —
  density falls to 0.00 of its fresh value with expiry-on-read and stays at 1.00
  (unchanged) when the published read path is reproduced, failing the assert.
  The same lazy expiry also leaks memory, quantified in
  [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

**Three are code-only defects that do not affect the paper's results:**

- **Findings E, G, H** — the LSH cell code discards information (angular: only Hamming weight
  survives; Euclidean: the k hashes are summed, collapsing 4,000 points into 101 cells), and the
  L2 ground-truth helper omits an exponent.

These three are **latent**: the paper sets the concatenation parameter to 1 for all
experiments, and at `k=1` all three are inert. They break only at `k>1` — the regime the
LSH-amplification argument is actually about. We fixed them because we intend to use
`k>1`. **The paper's published numbers stand**, and we make no claim otherwise.

The distinction matters and we keep it throughout: A and F are corrections to the
*published algorithm* at its own settings; E, G and H are bugs in the *reference code*
that its own experiments never triggered. We have not attempted to quantify what A and F
would change in the paper's reported figures, and so we do not assert anything about them.

The three validation tiers (see `CLAUDE.md` §9) are: exponential-histogram unit correctness;
un-windowed parity against plain RACE and full-stream brute force; and windowed accuracy
against brute-force last-N KDE within the paper's own theoretical bound.

## Getting started

The sketch on its own needs only numpy:

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest
```

### The monitoring stack

```bash
make env      # copy .env.example to .env, then edit the placeholders
make data     # download MetroPT-3 and convert to Parquet (~208 MB, not committed)
make up       # Kafka, consumer, Prometheus, Grafana, Alertmanager, Postgres
make anomaly  # replay the stream with a synthetic fault injected
```

`.env` holds the local Postgres and Grafana credentials and is **not committed** —
only [`.env.example`](.env.example) is. Nothing has a hardcoded password: compose
refuses to start without the variables set, and the Python entry points raise a
message naming `.env.example` rather than falling back to a guess.

Grafana on <http://localhost:3000> (anonymous access), Prometheus on `:9090`,
Alertmanager on `:9093`. `make down` stops everything.

### Agent access (MCP)

```bash
make alerts      # load detected episodes into Postgres
make mcp-verify  # exercise the three MCP tools end to end
make mcp-serve   # run the MCP server on stdio
```

## Layout

```
sketch/     the sketch itself: exponential histogram, LSH, SW-AKDE, RACE baseline, brute-force oracle
streaming/  producer, consumer, feature extraction, data-quality checks, scoring
docker/     Dockerfile and the Prometheus / Grafana / Alertmanager configuration
scripts/    dataset download, throughput benchmark
tests/      the three validation tiers plus streaming component tests
docs/       reference audit, performance log, data findings, and the source paper
CLAUDE.md   full project brief: novelty framing, findings, build plan, tech-stack rationale
```

## Data

MetroPT-3 (UCI): metro-train Air Production Unit sensors, 1,516,948 readings over
Feb–Sep 2020, 15 sensor channels, with four documented air-leak failures in a separate
maintenance report. The dataset is **not committed** — run `make data`.

Two things measurement contradicted, both detailed in
[`docs/DATA_NOTES.md`](docs/DATA_NOTES.md): the sampling rate is **0.1 Hz (every 10s), not
1 Hz** as the documentation says, and normal density is hugely variable because the
compressor cycles — enough that a naive z-score misses a 9× density collapse entirely.

## References

1. Danait, Das, Bhore. *Sublinear Sketches for ANN and KDE.* arXiv:2510.23039, 2025.
2. Coleman & Shrivastava. *Sub-linear RACE sketches for A-KDE on streaming data.* WWW 2020.
3. Datar, Gionis, Indyk, Motwani. *Maintaining stream statistics over sliding windows.*
   SIAM J. Comput., 2002.
4. Veloso et al. *The MetroPT dataset for predictive maintenance.* Scientific Data, 2022.
