import math
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from sketch import native
from streaming.failures import FAILURES, failure_at
from streaming.features import (
    ANALOG_COLUMNS,
    RollingDutyCycle,
    WarmupStandardizer,
    extract,
)
from streaming.quality import QualityMonitor
from streaming.scoring import RollingAnomalyScorer


def _reading(**overrides) -> dict:
    record = {column: 1.0 for column in ANALOG_COLUMNS}
    record["Oil_temperature"] = 60.0
    record.update(overrides)
    return record


class TestFeatures:
    def test_extract_pulls_columns_in_order(self):
        record = _reading(TP2=2.0, Motor_current=4.0)
        values = extract(record)
        assert values.shape == (len(ANALOG_COLUMNS),)
        assert values[ANALOG_COLUMNS.index("TP2")] == 2.0
        assert values[ANALOG_COLUMNS.index("Motor_current")] == 4.0

    def test_missing_and_unparseable_fields_become_nan(self):
        record = _reading(TP2="not-a-number")
        del record["H1"]
        values = extract(record)
        assert np.isnan(values[ANALOG_COLUMNS.index("TP2")])
        assert np.isnan(values[ANALOG_COLUMNS.index("H1")])
        assert not np.isnan(values[ANALOG_COLUMNS.index("TP3")])


class TestRollingDutyCycle:
    def test_reports_fraction_asserted_before_the_window_fills(self):
        duty = RollingDutyCycle(n_channels=2, window=4)
        assert duty.update(np.array([1.0, 0.0])).tolist() == [1.0, 0.0]
        assert duty.update(np.array([0.0, 0.0])).tolist() == [0.5, 0.0]

    def test_old_readings_leave_the_window(self):
        duty = RollingDutyCycle(n_channels=1, window=3)
        for _ in range(3):
            duty.update(np.array([1.0]))
        assert duty.update(np.array([0.0]))[0] == pytest.approx(2 / 3)
        assert duty.update(np.array([0.0]))[0] == pytest.approx(1 / 3)
        assert duty.update(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_matches_a_brute_force_rolling_mean(self):
        rng = np.random.default_rng(0)
        window = 25
        duty = RollingDutyCycle(n_channels=3, window=window)
        history: list[np.ndarray] = []

        for _ in range(200):
            values = (rng.random(3) < 0.4).astype(float)
            history.append(values)
            expected = np.vstack(history[-window:]).mean(axis=0)
            assert np.allclose(duty.update(values), expected)

    def test_constant_channel_gives_constant_duty(self):
        duty = RollingDutyCycle(n_channels=2, window=10)
        for _ in range(50):
            result = duty.update(np.array([1.0, 0.0]))
        assert result.tolist() == [1.0, 0.0]


class TestWarmupStandardizer:
    def test_not_fitted_until_warmup_reached(self):
        standardizer = WarmupStandardizer(n_features=3, warmup=10)
        for _ in range(9):
            standardizer.observe(np.ones(3))
        assert not standardizer.fitted
        standardizer.observe(np.ones(3))
        assert standardizer.fitted

    def test_transform_standardizes_to_roughly_zero_mean_unit_scale(self):
        rng = np.random.default_rng(0)
        standardizer = WarmupStandardizer(n_features=3, warmup=500)
        sample = rng.normal(loc=[10.0, -4.0, 100.0], scale=[2.0, 0.5, 25.0], size=(500, 3))
        for row in sample:
            standardizer.observe(row)

        transformed = np.vstack([standardizer.transform(row) for row in sample])
        assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-9)
        assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-9)

    def test_constant_channel_does_not_divide_by_zero(self):
        standardizer = WarmupStandardizer(n_features=2, warmup=10)
        for _ in range(10):
            standardizer.observe(np.array([5.0, 0.0]))
        out = standardizer.transform(np.array([5.0, 0.0]))
        assert np.all(np.isfinite(out))

    def test_statistics_freeze_after_warmup(self):
        # Deliberate: a rolling scaler would absorb a developing fault into its
        # own baseline and hide the drift we are trying to detect.
        standardizer = WarmupStandardizer(n_features=1, warmup=5)
        for _ in range(5):
            standardizer.observe(np.array([0.0]))
        frozen_mean = standardizer.mean.copy()

        for _ in range(1000):
            standardizer.observe(np.array([500.0]))
        assert np.array_equal(standardizer.mean, frozen_mean)
        # A large excursion must still read as far from normal.
        assert abs(standardizer.transform(np.array([500.0]))[0]) > 100


class TestQualityMonitor:
    def test_clean_record_reports_nothing(self):
        monitor = QualityMonitor(ANALOG_COLUMNS)
        values = extract(_reading())
        report = monitor.check(_reading(), values, datetime(2020, 2, 1, 0, 0, 0))
        assert report.clean
        assert report.usable

    def test_missing_field_makes_record_unusable(self):
        monitor = QualityMonitor(ANALOG_COLUMNS)
        record = _reading()
        del record["TP2"]
        report = monitor.check(record, extract(record), None)
        assert "TP2" in report.missing_fields
        assert not report.usable

    def test_out_of_range_is_flagged_but_still_usable(self):
        # Implausible values are reported, not silently dropped: a sensor
        # reading far outside its envelope may itself be the fault.
        monitor = QualityMonitor(ANALOG_COLUMNS)
        record = _reading(Oil_temperature=5000.0)
        report = monitor.check(record, extract(record), None)
        assert "Oil_temperature" in report.out_of_range
        assert report.usable
        assert not report.clean

    def test_nominal_10s_cadence_is_not_a_gap(self):
        # Regression: MetroPT-3 samples every 10s (0.1 Hz), not the 1 Hz its
        # documentation claims. A 5s threshold flagged literally every reading
        # as a gap and kept the data-quality alert permanently firing.
        monitor = QualityMonitor(ANALOG_COLUMNS)
        values = extract(_reading())
        start = datetime(2020, 2, 1, 0, 0, 0)
        for step in range(20):
            when = start + timedelta(seconds=10 * step)
            report = monitor.check(_reading(), values, when)
            assert report.gap_seconds == 0.0
            assert report.clean

    def test_detects_timestamp_regression_duplicate_and_gap(self):
        monitor = QualityMonitor(ANALOG_COLUMNS)
        values = extract(_reading())
        base = datetime(2020, 2, 1, 0, 0, 0)

        monitor.check(_reading(), values, base)
        assert monitor.check(_reading(), values, base).duplicate_timestamp
        assert monitor.check(_reading(), values, datetime(2020, 1, 1)).timestamp_regression

        monitor.check(_reading(), values, datetime(2020, 2, 1, 1, 0, 0))
        gapped = monitor.check(_reading(), values, datetime(2020, 2, 1, 1, 5, 0))
        assert gapped.gap_seconds == pytest.approx(300.0)


class TestRollingAnomalyScorer:
    def _scorer(self, **kwargs) -> RollingAnomalyScorer:
        defaults = {"min_history": 200, "smoothing": 40}
        return RollingAnomalyScorer(**{**defaults, **kwargs})

    def test_returns_zero_before_enough_history(self):
        scorer = self._scorer()
        assert scorer.score(10.0) == 0.0

    def test_rejects_history_too_short_for_the_smoothing_window(self):
        # Guards the startup false-positive: while the EWMA is still converging
        # the series trends smoothly, so its std is unrepresentatively small.
        with pytest.raises(ValueError):
            RollingAnomalyScorer(min_history=10, smoothing=40)

    def test_steady_density_scores_near_zero(self):
        rng = np.random.default_rng(0)
        scorer = self._scorer()
        for _ in range(400):
            score = scorer.score(100.0 + rng.normal(scale=1.0))
        assert score < 3.0

    def test_sustained_drop_beats_cyclic_noise(self):
        # The case measured on MetroPT-3: the air compressor cycles, so normal
        # density swings widely (there, median 42.7 with std 22.0). A 6x drop
        # reads as only ~1.7 sigma against that spread, so scoring raw density
        # misses it entirely. Smoothing averages the cycle out while preserving
        # a sustained shift.
        scorer = self._scorer()
        cycle = lambda i: 42.0 + 22.0 * math.sin(i / 3.0)  # noqa: E731

        for i in range(600):
            scorer.score(cycle(i))

        # A sustained collapse to the injected-fault level.
        for i in range(120):
            score = scorer.score(7.0 + 2.0 * math.sin(i / 3.0))
        assert score > 3.0

    def test_single_cycle_trough_is_not_an_anomaly(self):
        # The flip side: an ordinary low point in the compressor cycle must not
        # trip the alert, or the detector is useless on a cycling asset.
        scorer = self._scorer()
        cycle = lambda i: 42.0 + 22.0 * math.sin(i / 3.0)  # noqa: E731
        for i in range(600):
            score = scorer.score(cycle(i))
            assert score < 3.0

    def test_separates_a_drop_in_heavy_tailed_density(self):
        # The decisive finding from scripts/tune_detector.py: normal density on
        # MetroPT-3 spans nearly three orders of magnitude (measured 1.0-528,
        # median 81). Scored on a linear scale the robust spread is so wide that
        # a 9x collapse peaked at 1.4 and no smoothing value fixed it. Scoring
        # log density is what makes the same fault separable.
        rng = np.random.default_rng(0)
        scorer = self._scorer()

        # Log-normal normal operation, matching the measured shape.
        for _ in range(1200):
            normal_score = scorer.score(float(np.exp(rng.normal(loc=4.4, scale=0.9))))

        for _ in range(400):
            fault_score = scorer.score(float(np.exp(rng.normal(loc=2.2, scale=0.7))))

        assert normal_score < 1.0
        assert fault_score > 2.0
        # Separation is what the alert threshold is placed inside.
        assert fault_score > 2 * max(normal_score, 0.1)

    def test_density_spike_is_not_an_anomaly(self):
        # Only drops matter -- a region becoming denser is not a fault signal.
        rng = np.random.default_rng(0)
        scorer = self._scorer()
        for _ in range(400):
            scorer.score(100.0 + rng.normal(scale=1.0))
        for _ in range(50):
            score = scorer.score(1000.0)
        assert score == 0.0

    def test_constant_density_does_not_divide_by_zero(self):
        scorer = self._scorer()
        for _ in range(200):
            score = scorer.score(42.0)
        assert score == 0.0


class TestSketchFactory:
    """The Phase 5 backend wiring.

    `build_sketch` is the pipeline's only path to a sketch, and it has two kernel
    branches -- so a backend threaded through one and not the other would leave
    the deployed consumer silently on the Python core.
    """

    def _build(self, monkeypatch, backend: str, kernel: str = "euclidean"):
        from streaming import sketch_factory

        # SETTINGS is a frozen dataclass, so replace the whole object rather than
        # trying to set an attribute on it.
        monkeypatch.setattr(
            sketch_factory,
            "SETTINGS",
            replace(sketch_factory.SETTINGS, sketch_backend=backend),
        )
        return sketch_factory.build_sketch(kernel=kernel, dim=len(ANALOG_COLUMNS))

    @pytest.mark.parametrize("kernel", ["euclidean", "angular"])
    def test_python_backend_is_honoured_for_both_kernels(self, monkeypatch, kernel):
        assert self._build(monkeypatch, "python", kernel).backend == "python"

    @pytest.mark.parametrize("kernel", ["euclidean", "angular"])
    def test_auto_resolves_to_whatever_was_built(self, monkeypatch, kernel):
        # `auto` must never fail: it is the pipeline default, and a container
        # built without a compiler still has to run.
        sketch = self._build(monkeypatch, "auto", kernel)
        assert sketch.backend == ("native" if native.AVAILABLE else "python")

        # And the sketch it returns has to actually work.
        point = np.zeros(len(ANALOG_COLUMNS))
        for t in range(1, 21):
            sketch.update(point, t)
        assert sketch.query(point, 20) > 0
        assert sketch.cell_count > 0

    def test_unknown_backend_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="backend"):
            self._build(monkeypatch, "cpp")


class TestFailureGroundTruth:
    def test_four_documented_failures(self):
        assert len(FAILURES) == 4

    def test_intervals_are_ordered_and_well_formed(self):
        for event in FAILURES:
            assert event.start < event.end
        starts = [event.start for event in FAILURES]
        assert starts == sorted(starts)

    def test_lookup_inside_and_outside_a_failure(self):
        first = FAILURES[0]
        assert failure_at(first.start) is first
        assert failure_at(first.end) is first
        assert failure_at(datetime(2020, 3, 1)) is None
