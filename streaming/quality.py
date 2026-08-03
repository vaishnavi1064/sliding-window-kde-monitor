"""Data-quality checks on the incoming sensor stream.

The "data-quality" half of the monitor. These are cheap, per-record structural
checks -- distinct from the density-based anomaly detection, which is about the
sensor values being *unusual* rather than *malformed*. Both feed the dashboard.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from streaming.features import PLAUSIBLE_RANGES


@dataclass
class QualityReport:
    missing_fields: list[str] = field(default_factory=list)
    out_of_range: list[str] = field(default_factory=list)
    timestamp_regression: bool = False
    duplicate_timestamp: bool = False
    gap_seconds: float = 0.0

    @property
    def usable(self) -> bool:
        """Whether the record can be fed to the sketch at all."""
        return not self.missing_fields

    @property
    def clean(self) -> bool:
        return (
            not self.missing_fields
            and not self.out_of_range
            and not self.timestamp_regression
            and not self.duplicate_timestamp
        )


class QualityMonitor:
    """Stateful across records, so it can spot timestamp problems and gaps.

    Note that timestamp anomalies do not affect the sketch's own clock: the
    consumer drives the sketch with a monotonic per-event counter precisely so
    that irregular wall-clock timestamps cannot corrupt window expiry
    (CLAUDE.md Finding B). These checks report on data health only.
    """

    # MetroPT-3 samples every 10s (0.1 Hz), not the 1 Hz its documentation
    # suggests -- measured over all 1,516,948 readings: median gap 10.0s, p95
    # 10.0s, p99 12.0s, no duplicates and no regressions. The threshold is 3x
    # nominal so ordinary jitter is not reported as a gap.
    NOMINAL_INTERVAL_SECONDS = 10.0
    GAP_THRESHOLD_SECONDS = 3 * NOMINAL_INTERVAL_SECONDS

    def __init__(self, columns: tuple[str, ...]):
        self.columns = columns
        self.previous_timestamp: datetime | None = None

    def check(self, record: dict, values: np.ndarray, when: datetime | None) -> QualityReport:
        report = QualityReport()

        for column, value in zip(self.columns, values):
            if np.isnan(value):
                report.missing_fields.append(column)
                continue
            bounds = PLAUSIBLE_RANGES.get(column)
            if bounds is not None and not (bounds[0] <= value <= bounds[1]):
                report.out_of_range.append(column)

        if when is not None:
            previous = self.previous_timestamp
            if previous is not None:
                delta = (when - previous).total_seconds()
                if delta < 0:
                    report.timestamp_regression = True
                elif delta == 0:
                    report.duplicate_timestamp = True
                elif delta > self.GAP_THRESHOLD_SECONDS:
                    report.gap_seconds = delta
            self.previous_timestamp = when

        return report
