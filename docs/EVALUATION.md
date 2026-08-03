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

## Results

All figures from the full 1,516,948-reading replay, `rows=400`, `k=3`,
`window=3600`, seven analog channels.

### 1. The sketch matches exact windowed KDE almost exactly

Compared at equal alerting fractions:

| alerting | swakde events / p | exact events / p | race events / p |
|---|---|---|---|
| 20% | 4/4, p=0.567 | 4/4, p=0.585 | 4/4, p=0.495 |
| 10% | 4/4, p=0.290 | 4/4, p=0.296 | 4/4, p=0.172 |
| **5%** | **4/4, p=0.050** | **4/4, p=0.049** | 3/4, p=0.284 |
| 2% | 2/4, p=0.194 | 2/4, p=0.177 | 2/4, p=0.315 |
| 1% | 2/4, p=0.018 | 2/4, p=0.018 | 2/4, p=0.151 |

SW-AKDE tracks exact KDE to within noise at every operating point — 0.050 versus
0.049 at the 5% budget. **Approximation costs essentially nothing in detection
quality**, which is the positive result for the engineering lever.

Un-windowed RACE is meaningfully worse where it matters: at the 5% budget it
finds 3/4 rather than 4/4, and its p-values never get close. Sliding-window
semantics do buy something real here, which supports the source paper's claim.

### 2. It detects at onset; it does not predict

The prediction horizon changes the story completely, so it has to be reported
rather than chosen:

| horizon | events | mean lead | p | per-event lead |
|---|---|---|---|---|
| 3h | 4/4 | −0.6h | **0.004** | −2, −0, −0, −0 |
| 6h | 4/4 | −0.6h | 0.009 | −2, −0, −0, −0 |
| 12h | 4/4 | −0.6h | 0.024 | −2, −0, −0, −0 |
| 24h | 4/4 | +10.1h | 0.049 | +19, +22, −0, −0 |
| 48h | 4/4 | +27.2h | 0.135 | +19, +43, +47, −0 |

At a tight 3-hour horizon the detector finds all four failures with p = 0.004 —
clearly better than chance. But the lead times are **≈0**: it fires *as* each
failure begins. The large positive leads only appear once the horizon is widened,
and they vanish at 3h, which means they are the wider window catching unrelated
alarms rather than genuine early warning.

**This is a detection system, not a prediction system, on this data.** The
"+17h mean lead" that a naive reading of the 24h row would support is not
supported once chance is accounted for.

Caveat: this table scans 5 horizons x 5 thresholds, so quoting the best cell is
multiple comparisons over four events. The low values cluster consistently at
tight horizons rather than appearing in one lucky cell, which is more
reassuring than a single p, but strict Bonferroni over 25 cells (α = 0.002)
would not pass.

### 3. Why detection is limited: half the failures are invisible in these channels

`scripts/diagnose_failures.py` measures each channel's standardized shift in the
24 hours before each failure, against normal operation:

| failure | largest analog effect | verdict |
|---|---|---|
| failure-1 | Oil_temperature −1.91σ | clearly visible |
| failure-2 | Oil_temperature +0.62σ | barely visible |
| failure-3 | Oil_temperature +0.24σ | not visible |
| failure-4 | Oil_temperature +1.67σ | clearly visible |

Two of the four failures have essentially no analog signature beforehand. No
density-based detector on these channels can predict them, because the
information is not there.

The eight **digital** channels do shift (`Oil_level` 0.901 → 1.000,
`Caudal_impulses` 0.935 → 1.000, `DV_eletric` 0.143 → 0.024 or 0.393), so we
added their rolling duty cycles as features. **It made things worse** — best p
went from 0.108 to 0.193. Going from 7 to 15 dimensions diluted the density
estimate faster than the extra signal helped, which is the ordinary curse of
dimensionality for KDE. Reported as a failed attempt rather than quietly
dropped.

Honesty note: those duty-cycle features were chosen *after* inspecting the four
events. With n=4 that is a real overfitting risk, and it is one reason not to
keep iterating on feature selection here.

### 4. The uncomfortable one: at this dimensionality the sketch costs more memory than storing the window

The sketch exists so you never have to hold the window. But its footprint is
independent of dimension while exact storage grows with it, so there is a
crossover — and MetroPT sits on the wrong side of it.

Measured (`make memcheck-rows`), `window=3600`, `dim=7`:

| rows | live cells | sketch | vs exact | crossover dimension |
|---:|---:|---:|---:|---:|
| 50 | 7,398 | 5.2 MB | 26x | 179 |
| 100 | 15,383 | 12.0 MB | 60x | 418 |
| 200 | 29,849 | 23.6 MB | 117x | 819 |
| 400 | 60,330 | 47.5 MB | 236x | 1,650 |

Exact windowed KDE holds 3600 x 7 floats — **0.20 MB**. The sketch needs
5–48 MB for the same accuracy.

So on this workload the sketch is **26–236x more expensive than the thing it is
supposed to replace**, while detecting no better. Its advantage needs roughly
180 dimensions (at 50 rows) to 1,650 (at 400 rows) before storing the raw window
becomes the costlier option.

That is consistent with the source paper's own experiments, which used 103-,
200- and 384-dimensional data — comfortably inside the useful regime. A
seven-channel sensor feed is not. **The method is sound; this application is
outside the regime where it pays off.**

This does not undo the engineering work — the port is correct, faster, and
bounded, and it uncovered eight real defects. But an honest application study
has to report that the algorithm was applied to a problem its central advantage
does not address, and that a far simpler exact computation would be the right
engineering choice for MetroPT specifically.

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
