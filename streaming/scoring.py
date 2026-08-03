"""Turn a density estimate into an anomaly score.

Deliberately simple for Phase 2 -- enough to drive a dashboard and an alert.
The evaluated detector (thresholds, lead time, baselines against the four real
failures) is Phase 3 work.

The signal is a *drop* in density: the sketch answers "how many of the recent
readings look like this one?", so a reading in a region that has gone sparse is
the anomalous case. That is exactly what Finding F broke in the reference --
without expiry-on-read a quiet region's density never decays and this score
could never fire.

Every design choice below was measured with `scripts/tune_detector.py` against
real MetroPT-3 data rather than guessed; see docs/DATA_NOTES.md.
"""

import math
import statistics
from collections import deque

# Scale factor making MAD a consistent estimator of standard deviation for
# normally distributed data, so thresholds stay interpretable as "sigmas".
MAD_TO_SIGMA = 1.4826

# Floor on the robust scale, in log-density units. Without it, a stretch where
# the smoothed series barely moves drives MAD towards zero and the score towards
# infinity: measured on real data, 999 samples reached 5.5e6 "sigmas", which
# wrecked every quantile-based threshold above the 99th percentile. A deviation
# smaller than this in log space means a sub-0.1% density difference, which is
# not a resolvable signal regardless of how quiet the baseline was.
MIN_SCALE = 1e-3

# Scores above this are all equally "certainly anomalous", and letting them run
# to six figures only destabilises quantiles and dashboards.
MAX_SCORE = 100.0


class RollingAnomalyScorer:
    """Scores smoothed log-density against its own recent distribution.

    Three design points, each forced by what the data actually looks like:

    1. **Work in log space.** Density on this asset spans nearly three orders of
       magnitude during *normal* operation (measured: 1.0 to 528, median 81).
       It is a positive, multiplicative quantity, so on a linear scale the
       robust spread is enormous and even a 9x collapse scores under 1.5 sigma
       at any smoothing. Taking logs makes the distribution roughly symmetric
       and lifts the same fault to a clearly separated signal.

    2. **Smooth before scoring, and smooth generously.** The air compressor
       cycles, so density swings widely at high frequency while a developing
       fault is a sustained shift. Short smoothing produces a taller peak but it
       lasts only seconds and sits close to normal excursions; longer smoothing
       trades peak height for a signal that is both well separated and long
       enough to satisfy an alert's `for` duration. Measured separation between
       fault peak and worst normal excursion: 1.6x at smoothing=10 (5s long)
       versus 3.5x at smoothing=80 (87s long). Persistence wins.

    3. **Use a robust baseline (median and MAD), not mean and standard
       deviation.** A sustained fault's own samples land in the rolling history
       and drag the mean toward the anomaly while inflating the spread, so a
       mean-based score decays exactly when the fault persists. The median
       tolerates up to 50% contamination before it moves.

    Returns robust sigmas below the recent median, clamped at zero -- unusually
    *high* density is not a fault signal here.

    The baseline deliberately spans far more readings than the sketch's own
    window. The signal is inherently transient: once a new regime has filled the
    sliding window it *becomes* the normal and density recovers, so the score is
    elevated for roughly `window_size` readings after a shift and the baseline
    must remember normal operation for longer than that to see it.
    """

    def __init__(
        self,
        history: int = 2000,
        min_history: int = 400,
        smoothing: int = 80,
    ):
        if min_history < 5 * smoothing:
            # While the EWMA is still converging from its first sample the
            # series trends smoothly, so its spread is unrepresentatively small
            # and every reading looks significant. Requiring several time
            # constants of history avoids scoring noise as an anomaly at
            # startup.
            raise ValueError("min_history must be at least 5x smoothing")

        self.history: deque[float] = deque(maxlen=history)
        self.min_history = min_history
        self.alpha = 2.0 / (smoothing + 1.0)
        self.smoothed: float | None = None

    @property
    def smoothed_density(self) -> float | None:
        """The smoothed value back on the density scale, for display."""
        return None if self.smoothed is None else math.expm1(self.smoothed)

    def score(self, density: float) -> float:
        value = math.log1p(max(density, 0.0))
        self.smoothed = (
            value
            if self.smoothed is None
            else self.alpha * value + (1.0 - self.alpha) * self.smoothed
        )
        self.history.append(self.smoothed)

        if len(self.history) < self.min_history:
            return 0.0

        median = statistics.median(self.history)
        mad = statistics.median([abs(v - median) for v in self.history])
        scale = max(MAD_TO_SIGMA * mad, MIN_SCALE)

        return min(max(0.0, (median - self.smoothed) / scale), MAX_SCORE)
