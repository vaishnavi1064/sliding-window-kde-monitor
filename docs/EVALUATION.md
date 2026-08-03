# Evaluation methodology

How the detector is judged against MetroPT-3, and why it is judged that way.
Results are produced by `scripts/run_evaluation.py` and `scripts/compare_methods.py`.

## The core difficulty: four positives

MetroPT-3 documents exactly **four** air-leak failures, in a maintenance report
separate from the sensor data (there is no label column). Four is a very small
number of positives, and it shapes everything below.

**Precision and recall are not reported.** With four positives, one event either
way moves recall by 25%, and no confidence interval worth quoting can be placed
around either figure. Quoting "recall = 0.75" would imply a precision the data
cannot support. Papers do report such numbers on this dataset; we think that is
a mistake and decline to repeat it.

What four events *can* support:

| metric | why it survives n=4 |
|---|---|
| **per-event detection** | four separate yes/no answers, each individually meaningful |
| **detection lead time** | measured per event; the actual value proposition of predictive maintenance |
| **false-alarm rate** | denominated in *days of normal operation*, so it does not depend on the positive count at all |

## Prediction horizon

An alert that fires once a failure is already under way is close to worthless.
Each failure therefore gets a **24-hour horizon** before its recorded start, and
an alert inside that horizon counts as detecting it. Lead time is measured from
the failure's recorded start, so it is positive for genuine early warning and
negative for an alert raised mid-failure.

24 hours is a judgement call: long enough to credit real early warning on faults
that develop over hours, short enough not to swallow the timeline. It is a
parameter of the evaluation, not of the detector, and the sensitivity of results
to it should be reported rather than hidden.

## Alarms are episodes, not samples

A sustained fault puts thousands of consecutive samples above threshold.
Counting each as an alarm would make any false-alarm figure meaningless, so
consecutive alerting samples within an hour of each other collapse into one
**episode** — which is what an operator actually sees.

One consequence worth knowing when reading a sweep: because alarms are episodes,
the false-alarm rate is **not monotone** in the threshold. A low threshold can
merge a noisy stretch into a single long episode while a higher one fragments
the same stretch into several. Treat a sweep as a map of operating points, not a
monotone trade-off curve. (Detection counts *are* monotone.)

## Calibrating against luck — the control that matters most

This is the check that makes any detection claim meaningful, and it changed our
reading of the results completely.

With a 24-hour horizon around each failure, a detector that alarms roughly once
a day lands inside a horizon **most of the time by construction**. Catching all
four failures then says nothing whatsoever about the detector.

So every operating point is calibrated against a null: place the *same number of
alarms* uniformly at random across the same timeline, and count how many
failures that catches. Repeated 2,000 times, this gives

- **chance** — failures a random detector with the same alarm budget would catch
- **p** — the fraction of random trials matching or beating the real detector

A low p at a *small* alerting fraction is the only combination that demonstrates
genuine detection. A high detection count on its own does not.

Concretely, the un-windowed RACE baseline detects 4/4 failures at threshold 1.3
— but it is alerting **17.1% of the time**, chance alone catches 3.21/4, and
**p = 0.399**. Reported without the control, "detects all four failures with
17 hours of lead time" would have been badly misleading.

## Comparing methods fairly

Methods produce scores on different scales — un-windowed RACE's counts grow
without bound, so its score distribution is nothing like the sketch's. Comparing
them at a *fixed threshold* compares nothing.

`scripts/compare_methods.py` therefore selects each method's threshold as a
**quantile of its own score distribution**, so all methods are compared at the
same alerting fraction: the same alarm budget an operator would tolerate. That,
plus the chance calibration at each point, is the comparison.

## Baselines

| baseline | what it isolates |
|---|---|
| **un-windowed RACE** | what sliding-window semantics actually buy. Same LSH geometry, integer counters, nothing ever expires. This is the comparison the source paper itself makes. |
| **exact windowed KDE** | what bounded memory costs. Brute-force sum over the last N readings — the quantity the sketch approximates, with no sketch involved. Uses O(N·d) memory and O(N) work per query, precisely what the sketch exists to avoid, so it is a reference point rather than a deployable alternative. |

### TAKDE was considered and dropped

CLAUDE.md lists TAKDE (Wang, Ding & Shahrampour, TPAMI 2023) as a bonus
baseline. On reading the reference implementation
(`TAKDE/takde_alg.py`), it turns out to be a **one-dimensional synthetic
benchmark script**, not a reusable implementation:

- window selection bins with 1-D `np.histogram` over a scalar batch;
- ground truth is assumed to be a known Gaussian, `dst[t] = (mean, std)`;
- the relative error is computed inline, against that assumed truth;
- the query loop is a triple-nested Python loop over queries x window x batch.

Running it on 7-dimensional sensor data requires choosing a multivariate
two-sample statistic for the window-selection step, which the method's reference
does not provide. Any weakness in the result would then reflect *our* choice
rather than the published method, which makes the comparison indefensible rather
than informative. Dropping it, and saying so, is more honest than shipping a
"TAKDE" that is substantially ours. Revisiting it properly is a scoped piece of
work in its own right.

## Reproducing

```bash
make data                                   # download MetroPT-3
python -m scripts.run_evaluation --method swakde
python -m scripts.run_evaluation --method race
python -m scripts.run_evaluation --method exact
python -m scripts.compare_methods
```

The expensive replay (1.5M readings) runs once per method and caches its score
series to Parquet; every threshold sweep afterwards is instant.
