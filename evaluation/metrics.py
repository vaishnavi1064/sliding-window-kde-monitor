"""Evaluation metrics for the anomaly detector.

**Why not precision/recall.** There are exactly four documented failures. A
precision or recall figure computed on four positives carries almost no
information -- one event either way swings recall by 25%, and no confidence
interval worth quoting can be placed around it. Reporting "recall = 0.75" would
imply a precision the data cannot support.

What four events *can* support, and what an operator actually asks:

  - **Per-event detection.** Did we catch each failure, yes or no, and how far
    in advance? Four separate answers, each individually meaningful.
  - **Detection lead time.** How long before the failure did the first alert
    fire? This is the whole value proposition of predictive maintenance.
  - **False-alarm rate.** Alarms per day across genuinely normal operation.
    Denominated in *time*, not in events, so it does not depend on the tiny
    positive count.

Together those three characterise the detector honestly. A threshold sweep over
them gives the operating-point trade-off that an ROC curve would, without
pretending to a precision estimate.

**Reading a sweep.** Because alarms are counted as *episodes* rather than
samples, the false-alarm rate is not guaranteed to fall as the threshold rises:
a low threshold can merge a noisy stretch into one long episode, while a higher
one fragments the same stretch into several short ones. Treat a sweep as a map
of operating points, not as a monotone curve. (Detections *are* monotone.)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from evaluation.labels import DEFAULT_HORIZON, LabelledWindow, labelled_windows


@dataclass
class EventDetection:
    name: str
    detected: bool
    first_alert: datetime | None = None
    lead_time: timedelta | None = None

    @property
    def lead_hours(self) -> float | None:
        return None if self.lead_time is None else self.lead_time.total_seconds() / 3600.0


@dataclass
class EvaluationResult:
    threshold: float
    events: list[EventDetection] = field(default_factory=list)
    false_alarms: int = 0
    normal_days: float = 0.0
    alerting_fraction: float = 0.0

    @property
    def detected_count(self) -> int:
        return sum(1 for e in self.events if e.detected)

    @property
    def false_alarms_per_day(self) -> float:
        return self.false_alarms / self.normal_days if self.normal_days > 0 else float("inf")

    @property
    def mean_lead_hours(self) -> float | None:
        leads = [e.lead_hours for e in self.events if e.lead_hours is not None]
        return float(np.mean(leads)) if leads else None


def _alert_episodes(
    timestamps: np.ndarray, alerting: np.ndarray, min_gap: timedelta
) -> list[tuple[datetime, datetime]]:
    """Collapse consecutive alerting samples into episodes.

    Counting raw samples above threshold would report one continuous fault as
    thousands of alarms, which makes any false-alarm figure meaningless. An
    operator sees one alert per episode, so that is the unit.
    """
    if not alerting.any():
        return []

    indices = np.flatnonzero(alerting)
    episodes: list[tuple[datetime, datetime]] = []
    start = indices[0]
    previous = indices[0]

    for index in indices[1:]:
        gap = pd.Timestamp(timestamps[index]) - pd.Timestamp(timestamps[previous])
        if gap > min_gap:
            episodes.append(
                (pd.Timestamp(timestamps[start]).to_pydatetime(),
                 pd.Timestamp(timestamps[previous]).to_pydatetime())
            )
            start = index
        previous = index

    episodes.append(
        (pd.Timestamp(timestamps[start]).to_pydatetime(),
         pd.Timestamp(timestamps[previous]).to_pydatetime())
    )
    return episodes


def evaluate(
    timestamps: pd.Series,
    scores: np.ndarray,
    threshold: float,
    horizon: timedelta = DEFAULT_HORIZON,
    episode_gap: timedelta = timedelta(hours=1),
    windows: list[LabelledWindow] | None = None,
) -> EvaluationResult:
    """Score a detector at one threshold."""
    windows = windows if windows is not None else labelled_windows(horizon)
    values = pd.to_datetime(pd.Series(timestamps)).to_numpy()

    # Only judge failures the data actually covers. Scoring a failure that lies
    # outside the replayed span as "missed" would understate the detector on any
    # subset of the dataset, which is exactly when one is iterating fastest.
    span_start = pd.Timestamp(values[0]).to_pydatetime()
    span_end = pd.Timestamp(values[-1]).to_pydatetime()
    windows = [
        w for w in windows if w.event.start <= span_end and w.event.end >= span_start
    ]

    alerting = scores > threshold

    episodes = _alert_episodes(values, alerting, episode_gap)

    result = EvaluationResult(threshold=threshold)
    result.alerting_fraction = float(alerting.mean()) if alerting.size else 0.0

    matched: set[int] = set()
    for start, _end in episodes:
        hit = next(
            (i for i, w in enumerate(windows) if w.contains(start)),
            None,
        )
        if hit is None:
            result.false_alarms += 1
        else:
            matched.add(hit)

    for index, window in enumerate(windows):
        if index not in matched:
            result.events.append(EventDetection(name=window.name, detected=False))
            continue
        # Earliest alert episode falling inside this window.
        first = min(start for start, _ in episodes if window.contains(start))
        result.events.append(
            EventDetection(
                name=window.name,
                detected=True,
                first_alert=first,
                lead_time=window.lead_time(first),
            )
        )

    # Normal time = total span minus every horizon+failure window.
    span = pd.Timestamp(values[-1]) - pd.Timestamp(values[0])
    covered = timedelta(0)
    for window in windows:
        start = max(window.horizon_start, pd.Timestamp(values[0]).to_pydatetime())
        end = min(window.event.end, pd.Timestamp(values[-1]).to_pydatetime())
        if end > start:
            covered += end - start
    result.normal_days = max(
        (span.to_pytimedelta() - covered).total_seconds() / 86400.0, 1e-9
    )
    return result


def chance_detection(
    timestamps: pd.Series,
    n_alarms: int,
    horizon: timedelta = DEFAULT_HORIZON,
    trials: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Calibrate a detection count against luck.

    This is the check that makes a "4/4 events detected" claim mean anything.
    With a 24-hour horizon around each failure, a detector that alarms roughly
    once a day will land inside a horizon most of the time *by construction* --
    catching every failure then says nothing about the detector.

    Places `n_alarms` episodes uniformly at random across the same timeline and
    measures how often that alone catches the events.

    Returns the per-trial detection counts; compare a real detector's count
    against them with `chance_p_value`.
    """
    windows = labelled_windows(horizon)
    values = pd.to_datetime(pd.Series(timestamps)).to_numpy()
    span_start = pd.Timestamp(values[0]).to_pydatetime()
    span_end = pd.Timestamp(values[-1]).to_pydatetime()
    windows = [
        w for w in windows if w.event.start <= span_end and w.event.end >= span_start
    ]
    if not windows or n_alarms <= 0:
        return 0.0, 1.0

    total_seconds = (span_end - span_start).total_seconds()
    bounds = np.array(
        [
            [
                (max(w.horizon_start, span_start) - span_start).total_seconds(),
                (min(w.event.end, span_end) - span_start).total_seconds(),
            ]
            for w in windows
        ]
    )

    rng = np.random.default_rng(seed)
    counts = np.empty(trials, dtype=int)
    for trial in range(trials):
        offsets = rng.uniform(0.0, total_seconds, size=n_alarms)
        hit = (offsets[:, None] >= bounds[None, :, 0]) & (
            offsets[:, None] <= bounds[None, :, 1]
        )
        counts[trial] = int(hit.any(axis=0).sum())

    return counts


def chance_p_value(counts: np.ndarray, observed: int) -> float:
    """Fraction of random-alarm trials matching or beating the observed count."""
    if isinstance(counts, float) or len(counts) == 0:
        return 1.0
    return float((counts >= observed).mean())


def sweep(
    timestamps: pd.Series,
    scores: np.ndarray,
    thresholds: np.ndarray,
    horizon: timedelta = DEFAULT_HORIZON,
) -> list[EvaluationResult]:
    windows = labelled_windows(horizon)
    return [
        evaluate(timestamps, scores, float(t), horizon, windows=windows) for t in thresholds
    ]


def format_result(result: EvaluationResult) -> str:
    lines = [
        f"threshold {result.threshold:.2f}: "
        f"{result.detected_count}/4 events, "
        f"{result.false_alarms_per_day:.2f} false alarms/day "
        f"({result.false_alarms} over {result.normal_days:.0f} normal days)"
    ]
    for event in result.events:
        if event.detected:
            lines.append(f"    {event.name}: detected, {event.lead_hours:+.1f}h lead")
        else:
            lines.append(f"    {event.name}: MISSED")
    return "\n".join(lines)
