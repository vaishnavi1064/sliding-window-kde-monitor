"""Ground-truth failure events for MetroPT-3.

MetroPT-3 ships no label column. The dataset's companion maintenance report
documents four air-leak failures, all "High stress" severity; labels have to be
derived by joining these intervals against the sensor timestamps.

Four positives is a very small evaluation set. Phase 3 should report per-event
detection and lead time plus a false-alarm rate over the normal stretches,
rather than a precision/recall figure computed on n=4 (see CLAUDE.md).
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FailureEvent:
    name: str
    start: datetime
    end: datetime
    kind: str = "air leak"

    def contains(self, when: datetime) -> bool:
        return self.start <= when <= self.end


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text)


FAILURES: tuple[FailureEvent, ...] = (
    FailureEvent("failure-1", _at("2020-04-18T00:00:00"), _at("2020-04-18T23:59:00")),
    FailureEvent("failure-2", _at("2020-05-29T23:30:00"), _at("2020-05-30T06:00:00")),
    FailureEvent("failure-3", _at("2020-06-05T10:00:00"), _at("2020-06-07T14:30:00")),
    FailureEvent("failure-4", _at("2020-07-15T14:30:00"), _at("2020-07-15T19:00:00")),
)


def failure_at(when: datetime) -> FailureEvent | None:
    for event in FAILURES:
        if event.contains(when):
            return event
    return None
