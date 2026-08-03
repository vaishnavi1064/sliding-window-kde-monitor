"""Derive evaluation labels from the MetroPT-3 maintenance report.

The dataset has no label column; the four documented air-leak failures live in a
separate report and are encoded in `streaming.failures`. Labels are produced by
joining those intervals against sensor timestamps.

The central modelling choice here is the **prediction horizon**. For predictive
maintenance, an alert that fires the moment a failure is already under way is
nearly worthless -- the value is in warning beforehand. So each failure gets a
horizon before its recorded start, and an alert inside that horizon counts as a
detection of that failure. Alerts outside every horizon-plus-failure span are
false alarms.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from streaming.failures import FAILURES, FailureEvent

# How far ahead of a recorded failure an alert still counts as predicting it.
# The MetroPT paper's own framing treats these as slowly developing air leaks,
# so hours rather than minutes; 24h is long enough to give a detector credit for
# genuine early warning without being so long it swallows normal operation.
DEFAULT_HORIZON = timedelta(hours=24)


@dataclass(frozen=True)
class LabelledWindow:
    """A failure plus the horizon in which an alert counts as predicting it."""

    event: FailureEvent
    horizon_start: datetime

    @property
    def name(self) -> str:
        return self.event.name

    def contains(self, when: datetime) -> bool:
        """True inside the horizon or the failure itself."""
        return self.horizon_start <= when <= self.event.end

    def lead_time(self, when: datetime) -> timedelta:
        """How far before the failure started this alert fired.

        Negative once the failure is already under way.
        """
        return self.event.start - when


def labelled_windows(
    horizon: timedelta = DEFAULT_HORIZON,
    failures: tuple[FailureEvent, ...] = FAILURES,
) -> list[LabelledWindow]:
    return [LabelledWindow(event=f, horizon_start=f.start - horizon) for f in failures]


def label_series(
    timestamps: pd.Series, horizon: timedelta = DEFAULT_HORIZON
) -> tuple[np.ndarray, np.ndarray]:
    """Label each timestamp.

    Returns (in_window, window_index) where `in_window` marks the horizon or
    failure span of some event, and `window_index` gives which one (-1 if none).
    """
    windows = labelled_windows(horizon)
    in_window = np.zeros(len(timestamps), dtype=bool)
    which = np.full(len(timestamps), -1, dtype=int)

    values = pd.to_datetime(timestamps).to_numpy()
    for index, window in enumerate(windows):
        mask = (values >= np.datetime64(window.horizon_start)) & (
            values <= np.datetime64(window.event.end)
        )
        in_window |= mask
        which[mask] = index

    return in_window, which


def normal_duration(
    timestamps: pd.Series, horizon: timedelta = DEFAULT_HORIZON
) -> timedelta:
    """Wall-clock span of the data that is neither failure nor horizon.

    The denominator for a false-alarm *rate* -- alarms per day of genuinely
    normal operation, which is the figure an operator actually cares about.
    """
    in_window, _ = label_series(timestamps, horizon)
    values = pd.to_datetime(timestamps)
    total = values.iloc[-1] - values.iloc[0]
    if not in_window.any():
        return total

    covered = timedelta(0)
    for window in labelled_windows(horizon):
        start = max(window.horizon_start, values.iloc[0].to_pydatetime())
        end = min(window.event.end, values.iloc[-1].to_pydatetime())
        if end > start:
            covered += end - start
    return total.to_pytimedelta() - covered
