# What the real data forced

Findings from actually streaming MetroPT-3 through the sketch, as opposed to
reading its documentation. Each one changed the code. These matter most for
Phase 3, where the detector gets designed and evaluated properly.

## 1. MetroPT-3 is 0.1 Hz, not 1 Hz

Both the dataset documentation and our own project brief said 1 Hz. Measured
across all 1,516,948 readings:

| statistic | value |
|---|---|
| median inter-reading gap | 10.0 s |
| mean | 12.1 s |
| p1 / p25 / p75 / p95 / p99 | 9 / 10 / 10 / 10 / 12 s |
| duplicate timestamps | 0 |
| timestamp regressions | 0 |

The span confirms it: 1,516,948 readings x 10 s ≈ 176 days, matching the
2020-02-01 → 2020-09-01 range. At 1 Hz the file would cover only 17.5 days.
(UCI's own page hedges — "collected at 1Hz, later described as 0.1Hz".)

Consequences, all now fixed in code:

- The data-quality gap threshold was 5 s, so **100% of readings were flagged as
  gaps** and the data-quality alert sat permanently firing. Now 30 s (3x
  nominal), with a regression test pinning the 10 s cadence as clean.
- `window_size = 3600` readings is **~10 hours** of asset time, not one hour.
- Real-time replay would take ~176 days, not ~17.5.

Worth noting for Finding B (the logical clock): this dataset turns out to have
**no duplicate or backwards timestamps**, so the wall-clock hazard is milder
here than feared. The logical clock is still the right call — it costs nothing
and the irregular 9–12 s jitter would otherwise leak into window semantics — but
we should not overstate the danger on this particular dataset.

## 2. Density spans three orders of magnitude, so linear-scale scoring cannot work

Measured offline over 30,000 readings with a synthetic 4x fault injected
(`scripts/tune_detector.py`, 2,441 density samples):

| | median | IQR | range |
|---|---|---|---|
| normal | 81.2 | 39.1 – 140.7 | **1.0 – 528.4** |
| injected fault | 9.3 | 5.0 – 18.2 | 1.0 – 62.2 |

The fault drops median density **9x** — an unmistakable signal. Yet the first
detector (robust z-score of raw density) peaked at **1.4** against a threshold
of 3, and *no amount of smoothing fixed it*: the ceiling across the whole sweep
was 1.41.

The cause is the shape of the distribution, not the sketch and not the
parameters. The Air Production Unit cycles on and off, and normal density
consequently spans nearly three orders of magnitude. Density is a positive,
multiplicative quantity, so on a linear scale its robust spread is so wide that
a 9x collapse is unremarkable.

Three changes, all in `streaming/scoring.py`, each chosen from the measured
sweep rather than intuition:

1. **Score in log space.** This is the one that mattered. `log1p` makes the
   distribution roughly symmetric; the same fault goes from 1.41 to 4.13 at
   short smoothing.
2. **Smooth generously (EWMA span 80).** There is a real trade-off between peak
   height and persistence, and persistence is what an alert needs:

   | smoothing | fault peak | worst normal | separation | time above threshold |
   |---:|---:|---:|---:|---:|
   | 10 | 4.13 | 2.57 | 1.6x | 5 s |
   | 20 | 3.69 | 2.81 | 1.3x | 5 s |
   | 40 | 2.89 | 2.39 | 1.2x | 4 s |
   | **80** | **2.44** | **0.70** | **3.5x** | **87 s** |
   | 160 | 2.04 | 0.78 | 2.6x | 104 s |

   Short smoothing gives the tallest spike, but it lasts seconds and sits close
   to ordinary normal excursions. Optimising peak height was the wrong target.
3. **Robust baseline (median + MAD).** A sustained fault's own samples enter the
   rolling history — roughly 17-18% of it here — dragging a mean toward the
   anomaly and inflating the spread, so a mean-based score decays exactly when
   the fault persists. The median tolerates up to 50% contamination.

**Thresholds follow from that table, not from taste:** warning at 1.3, critical
at 2.0, sitting between the worst normal excursion (0.70) and the fault peak
(2.44). The same numbers appear in `docker/prometheus/alerts.yml` and on the
dashboard.

**Scope honesty:** this is still a deliberately simple detector, adequate for
driving a dashboard and an alert, and these numbers come from a *synthetic*
injected fault, not a real one. Whether real air-leak failures produce a
comparable density signature is exactly what Phase 3 has to establish, against
the four documented events — and it may well need a different statistic again.
`scripts/tune_detector.py` is the harness that work should build on.

## 3. A sliding-window density signal is inherently transient

Once a new regime has filled the sliding window it *becomes* the normal, and
density recovers to baseline. So the elevated-score period lasts roughly
`window_size` readings after a shift, not for as long as the fault lasts.

This is correct behaviour for a sliding-window method rather than a defect, but
it has two practical consequences:

- The alert's `for:` duration has to be shorter than that span, and the span in
  wall-clock terms scales with replay speed. The Phase 2 demo replays at 60
  readings/s so the ~3600-reading span is ~60 s, comfortably above the 30 s
  `for`. On a live 0.1 Hz feed the same span is ~10 hours and `for: 5m` would be
  appropriate.
- The rolling baseline must remember normal operation for **longer** than that
  span, or it adapts away the very signal it is meant to detect. Hence
  `history = 2000` smoothed samples against a 3600-reading window.

Phase 3 should consider whether a *return* to normal after a fault (also a
novelty, also a density drop) needs distinguishing from fault onset.

## 4. Density is not comparable until the window has filled

A subtler version of the same problem. While the sliding window is still
filling, density ramps up from zero purely as an artefact of how much data the
window holds — nothing to do with the asset. Feeding that ramp to the scorer
poisons its baseline: the spread it measures is dominated by the ramp rather
than by real variation, which flattens the score during an actual fault.

The consumer now gates scoring on `clock >= window_size`, and exports
`swakde_window_full` so the dashboard can show it. This costs a longer startup
(warmup 2,000 readings, then 3,600 to fill the window, then the scorer's own
baseline) but the alternative is a detector that is quietly desensitised.

## Method note

Parameters here were chosen by replaying the real data offline through the real
sketch (`scripts/tune_detector.py`, seconds per configuration) rather than
through the Docker stack (minutes per configuration). That distinction mattered:
the first two attempts at this detector were tuned by intuition against slow
end-to-end runs, and both were wrong in ways the sweep exposed immediately —
the linear-scale ceiling, and the peak-versus-persistence trade-off.

