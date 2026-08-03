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
| 1 | Validated pure-Python sketch (angular kernel) | Done — 11 tests green |
| 1.5 | Euclidean kernel, profiling/vectorization | Next |
| 2 | Kafka streaming, Prometheus/Grafana/Alertmanager | Planned |
| 3 | Anomaly detector + MetroPT evaluation | Planned |
| 4 | MCP server | Planned |
| 5 | C++17 + pybind11 optimized core | Planned |
| 6 | Adaptive window size (research extension) | Stretch |

## Correctness

We validate bottom-up against ground truth that involves no sketch at all, never against the
reference implementation's outputs — because auditing that implementation turned up eight real
bugs, documented with evidence in [`docs/REFERENCE_NOTES.md`](docs/REFERENCE_NOTES.md). Some
trace back to the paper's own pseudocode rather than just the code:

- **Finding A** — every cell silently drops its first arrival.
- **Finding F** — cells that stop receiving data never expire, so their density is frozen
  forever. This is the one that matters most here: a region going quiet *is* the anomaly signal.
- **Findings E, G, H** — the LSH cell code discards information (angular: only Hamming weight
  survives; Euclidean: the k hashes are summed, collapsing 4,000 points into 101 cells), and the
  L2 ground-truth helper omits an exponent.

**These last three do not invalidate the paper's results.** It sets the concatenation parameter
to 1 for all experiments, where all three are inert. They are latent bugs that break at k>1 —
the regime the LSH-amplification argument is actually about. We fixed them because we intend to
use k>1; the published numbers stand.

The three validation tiers (see `CLAUDE.md` §9) are: exponential-histogram unit correctness;
un-windowed parity against plain RACE and full-stream brute force; and windowed accuracy
against brute-force last-N KDE within the paper's own theoretical bound.

## Getting started

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest
```

Or via `make test`.

## Layout

```
sketch/     the sketch itself: exponential histogram, LSH, SW-AKDE, RACE baseline, brute-force oracle
tests/      the three validation tiers
docs/       reference audit (REFERENCE_NOTES.md) and the source paper
CLAUDE.md   full project brief: novelty framing, findings, build plan, tech-stack rationale
```

## Data

MetroPT-3 (UCI): metro-train Air Production Unit sensors, 1.5M readings at 1 Hz over
Feb–Aug 2020, 15 sensor channels, with four documented air-leak failures in a separate
maintenance report. The dataset is **not committed**; a download script lands in Phase 2.

## References

1. Danait, Das, Bhore. *Sublinear Sketches for ANN and KDE.* arXiv:2510.23039, 2025.
2. Coleman & Shrivastava. *Sub-linear RACE sketches for A-KDE on streaming data.* WWW 2020.
3. Datar, Gionis, Indyk, Motwani. *Maintaining stream statistics over sliding windows.*
   SIAM J. Comput., 2002.
4. Veloso et al. *The MetroPT dataset for predictive maintenance.* Scientific Data, 2022.
