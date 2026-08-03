from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from evaluation.labels import DEFAULT_HORIZON, label_series, labelled_windows
from evaluation.metrics import evaluate, sweep
from streaming.failures import FailureEvent

# A small synthetic timeline so the tests do not depend on the real dataset.
BASE = datetime(2020, 1, 1)
ONE_EVENT = (
    FailureEvent("f1", datetime(2020, 1, 10), datetime(2020, 1, 10, 12)),
)


def _timeline(days: int = 20, step_minutes: int = 30) -> pd.Series:
    n = int(days * 24 * 60 / step_minutes)
    return pd.Series([BASE + timedelta(minutes=step_minutes * i) for i in range(n)])


def _windows():
    return [
        w
        for w in labelled_windows(horizon=timedelta(hours=24), failures=ONE_EVENT)
    ]


class TestLabels:
    def test_horizon_opens_before_the_failure(self):
        window = _windows()[0]
        assert window.horizon_start == datetime(2020, 1, 9)
        assert window.contains(datetime(2020, 1, 9, 1))       # inside horizon
        assert window.contains(datetime(2020, 1, 10, 6))      # during failure
        assert not window.contains(datetime(2020, 1, 8, 23))  # before horizon

    def test_lead_time_is_positive_before_failure_and_negative_during(self):
        window = _windows()[0]
        assert window.lead_time(datetime(2020, 1, 9, 12)) == timedelta(hours=12)
        assert window.lead_time(datetime(2020, 1, 10, 6)) == timedelta(hours=-6)

    def test_label_series_marks_only_the_window(self):
        timestamps = _timeline()
        in_window, which = label_series(timestamps, horizon=timedelta(hours=24))
        # Real FAILURES are in 2020-02..07, outside this synthetic January span.
        assert not in_window.any()
        assert (which == -1).all()


class TestEvaluate:
    def _run(self, scores: np.ndarray, timestamps: pd.Series, threshold: float = 1.0):
        return evaluate(
            timestamps, scores, threshold, episode_gap=timedelta(hours=1),
            windows=_windows(),
        )

    def test_detects_an_event_and_reports_lead_time(self):
        timestamps = _timeline()
        scores = np.zeros(len(timestamps))
        # Alert 12 hours before the failure starts.
        alert_at = datetime(2020, 1, 9, 12)
        scores[timestamps.searchsorted(alert_at)] = 5.0

        result = self._run(scores, timestamps)
        assert result.detected_count == 1
        assert result.events[0].detected
        assert result.events[0].lead_hours == pytest.approx(12.0, abs=0.6)
        assert result.false_alarms == 0

    def test_alert_outside_the_window_is_a_false_alarm(self):
        timestamps = _timeline()
        scores = np.zeros(len(timestamps))
        scores[timestamps.searchsorted(datetime(2020, 1, 3))] = 5.0

        result = self._run(scores, timestamps)
        assert result.detected_count == 0
        assert result.false_alarms == 1
        assert result.events[0].detected is False

    def test_a_continuous_fault_counts_as_one_alarm_not_thousands(self):
        # Without episode collapsing, a sustained excursion would report every
        # sample above threshold as a separate alarm and make the false-alarm
        # rate meaningless.
        timestamps = _timeline()
        scores = np.zeros(len(timestamps))
        start = timestamps.searchsorted(datetime(2020, 1, 3))
        scores[start : start + 200] = 5.0  # 100 hours of continuous alerting

        result = self._run(scores, timestamps)
        assert result.false_alarms == 1

    def test_separate_episodes_are_counted_separately(self):
        timestamps = _timeline()
        scores = np.zeros(len(timestamps))
        scores[timestamps.searchsorted(datetime(2020, 1, 3))] = 5.0
        scores[timestamps.searchsorted(datetime(2020, 1, 5))] = 5.0

        result = self._run(scores, timestamps)
        assert result.false_alarms == 2

    def test_false_alarm_rate_is_per_day_of_normal_operation(self):
        timestamps = _timeline(days=20)
        scores = np.zeros(len(timestamps))
        scores[timestamps.searchsorted(datetime(2020, 1, 3))] = 5.0

        result = self._run(scores, timestamps)
        # 20 days total minus the ~1.5-day window; one alarm over that span.
        assert 17 < result.normal_days < 20
        assert result.false_alarms_per_day == pytest.approx(
            1 / result.normal_days, rel=1e-6
        )

    def test_failures_outside_the_replayed_span_are_not_counted_as_missed(self):
        # Evaluating a subset of the dataset must not be penalised for failures
        # that subset does not contain -- otherwise every quick iteration looks
        # like a total detector failure.
        timestamps = _timeline(days=3)  # 2020-01-01..04; the event is 2020-01-10
        result = self._run(np.zeros(len(timestamps)), timestamps)
        assert result.events == []
        assert result.detected_count == 0

    def test_silent_detector_misses_everything_without_false_alarms(self):
        timestamps = _timeline()
        result = self._run(np.zeros(len(timestamps)), timestamps)
        assert result.detected_count == 0
        assert result.false_alarms == 0
        assert result.mean_lead_hours is None

    def test_raising_the_threshold_never_increases_detections(self):
        rng = np.random.default_rng(0)
        timestamps = _timeline()
        scores = rng.uniform(0, 3, size=len(timestamps))

        results = sweep(
            timestamps, scores, np.array([0.5, 1.0, 1.5, 2.0, 2.5]),
            horizon=timedelta(hours=24),
        )
        detections = [r.detected_count for r in results]
        assert detections == sorted(detections, reverse=True)

    def test_episode_counting_is_not_monotone_in_the_threshold(self):
        # A property worth knowing when reading a sweep, not a defect. Because
        # alarms are counted as episodes rather than samples, a *low* threshold
        # on a noisy score merges everything into one long episode (1 alarm),
        # while a *higher* threshold fragments the same noise into many isolated
        # blips (many alarms). So false-alarm rate can rise with the threshold.
        # Real detectors are not pure noise, but a sweep can still be
        # non-monotone here and should be read as an operating-point map rather
        # than a monotone trade-off curve.
        rng = np.random.default_rng(0)
        timestamps = _timeline()
        scores = rng.uniform(0, 3, size=len(timestamps))

        low, high = sweep(
            timestamps, scores, np.array([0.5, 2.5]), horizon=timedelta(hours=24)
        )
        assert low.false_alarms < high.false_alarms
