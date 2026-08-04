"""Tests for the MCP tool logic that do not need a live database.

The Postgres-backed queries are exercised by scripts/verify_mcp.py against a
real instance; what is unit-tested here is the logic an agent's answer actually
depends on -- the health verdict and the alert explanation -- since those are
what a user ends up reading.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from mcp_server.server import (
    DETECTOR_CAVEATS,
    _describe,
    alert_verdict,
    why_it_fired,
)
from mcp_server.store import Anomaly, health_status


def _anomaly(**overrides) -> Anomaly:
    defaults = dict(
        anomaly_id=1,
        asset_id="metropt-apu",
        started_at=datetime(2020, 4, 17, 5, 0),
        ended_at=datetime(2020, 4, 17, 8, 0),
        peak_score=4.2,
        threshold=2.8,
        method="swakde",
        matched_failure=None,
        lead_hours=None,
    )
    return Anomaly(**{**defaults, **overrides})


class TestHealthStatus:
    def test_no_recent_anomalies_is_healthy(self):
        assert health_status(0, None) == "healthy"

    def test_recent_low_score_anomaly_is_warning(self):
        assert health_status(2, _anomaly(peak_score=1.4)) == "warning"

    def test_recent_high_score_anomaly_is_critical(self):
        assert health_status(1, _anomaly(peak_score=2.5)) == "critical"

    def test_healthy_wins_even_if_an_old_alert_was_severe(self):
        # Status is about *recent* activity; a severe alert outside the 24h
        # window must not pin the asset to critical forever.
        assert health_status(0, _anomaly(peak_score=9.9)) == "healthy"


class TestAlertVerdict:
    def test_unmatched_alert_is_called_a_false_alarm(self):
        verdict = alert_verdict(_anomaly(matched_failure=None))
        assert "false alarm" in verdict
        # The rate must be quoted so an agent can convey how often this happens.
        assert "per day" in verdict

    def test_matched_alert_names_the_failure(self):
        verdict = alert_verdict(
            _anomaly(matched_failure="failure-1", lead_hours=18.6)
        )
        assert "failure-1" in verdict
        assert "18.6h before" in verdict

    def test_near_zero_lead_is_not_described_as_advance_warning(self):
        # The evaluation's central caveat: the detector fires at onset. Wording
        # that implied advance warning here would misrepresent it.
        verdict = alert_verdict(
            _anomaly(matched_failure="failure-3", lead_hours=-0.08)
        )
        assert "essentially at the failure's recorded start" in verdict
        assert "before" not in verdict

    def test_negative_lead_never_reads_as_positive_lead(self):
        verdict = alert_verdict(
            _anomaly(matched_failure="failure-4", lead_hours=-3.0)
        )
        assert "-3" not in verdict
        assert "essentially at" in verdict


class TestExplanation:
    def test_reports_how_far_past_threshold(self):
        text = why_it_fired(_anomaly(peak_score=5.6, threshold=2.8))
        assert "2.0x" in text

    def test_zero_threshold_does_not_divide_by_zero(self):
        text = why_it_fired(_anomaly(peak_score=1.0, threshold=0.0))
        assert "0.0x" in text

    def test_describes_a_density_drop_not_a_rise(self):
        # The signal is density falling. Describing it as a rise would invert
        # the detector's meaning.
        text = why_it_fired(_anomaly())
        assert "fell" in text
        assert "sparse" in text


class TestDescribe:
    def test_serialises_timestamps_and_rounds_scores(self):
        payload = _describe(_anomaly(peak_score=4.23456, threshold=2.80001))
        assert payload["started_at"] == "2020-04-17T05:00:00"
        assert payload["peak_score"] == 4.235
        assert payload["duration_hours"] == 3.0

    def test_unmatched_alert_reports_none_not_a_placeholder(self):
        payload = _describe(_anomaly(matched_failure=None))
        assert payload["coincided_with_documented_failure"] is None
        assert payload["lead_hours"] is None


class TestCaveats:
    def test_caveats_state_the_at_onset_limitation(self):
        # This text is what reaches a user through an agent, so the two findings
        # that most affect how a health verdict should be read are pinned here.
        assert "at onset rather than predicting" in DETECTOR_CAVEATS
        assert "not that nothing is developing" in DETECTOR_CAVEATS


class TestEpisodeLoading:
    def _series_with_a_late_density_collapse(self) -> pd.DataFrame:
        """Flat density, then a sustained collapse landing inside failure-1's horizon.

        Two constraints pull in opposite directions and both have to be met:
        the collapse must last longer than the scorer's smoothing span (80) for
        the EWMA to track it, yet be a small enough *fraction* of the series that
        a 5% quantile threshold sits below it rather than inside it. A long
        baseline satisfies both.
        """
        rng = np.random.default_rng(0)
        baseline, collapse = 3000, 120  # collapse is 3.8% of samples
        end = datetime(2020, 4, 17, 22, 0)  # failure-1's horizon opens 04-17 00:00
        total = baseline + collapse
        times = [end - timedelta(minutes=10 * (total - 1 - i)) for i in range(total)]
        density = np.concatenate([
            100.0 + rng.normal(scale=2.0, size=baseline),
            np.full(collapse, 4.0),
        ])
        return pd.DataFrame({"timestamp": times, "density": density})

    def test_loader_emits_episodes_and_matches_documented_failures(self):
        # The loader must not invent its own notion of an alert: an agent's view
        # has to agree with the report, so it reuses the evaluation's episode
        # collapsing and its failure labels.
        from scripts.load_alerts import episodes_for

        rows, threshold = episodes_for(self._series_with_a_late_density_collapse(), 0.05)

        assert rows, "a sustained density collapse should produce an episode"
        assert threshold > 0
        # The collapse sits inside failure-1's 24h horizon, so it must be
        # attributed rather than counted as a false alarm.
        matched = [r for r in rows if r["matched_failure"] == "failure-1"]
        assert matched, [r["matched_failure"] for r in rows]
        assert matched[0]["peak_score"] >= threshold
        # Positive lead: the episode starts before the failure's recorded start.
        assert matched[0]["lead_hours"] > 0

    def test_a_sustained_collapse_is_one_episode_not_many(self):
        from scripts.load_alerts import episodes_for

        rows, _ = episodes_for(self._series_with_a_late_density_collapse(), 0.05)

        # The 120 breaching samples are 10 minutes apart, inside the 1h episode
        # gap, so they must collapse into a single alert rather than 120.
        collapse_start = datetime(2020, 4, 17, 2, 0)
        during = [r for r in rows if r["started_at"] >= collapse_start]
        assert len(during) == 1, during
        assert during[0]["ended_at"] - during[0]["started_at"] >= timedelta(hours=15)

        # There is also an earlier, unrelated episode: right after the scorer
        # reaches min_history its EWMA is still converging, which produces a
        # short transient. Asserting on the total count would therefore be
        # asserting on that artefact rather than on episode collapsing. In the
        # real pipeline the consumer additionally gates scoring on the sliding
        # window being full, which suppresses it; here that gate is bypassed.
        assert len(rows) >= 1


def test_store_requires_a_dsn_but_does_not_connect_on_construction():
    # Constructing the store must not open a socket, or importing the server
    # would fail wherever Postgres is not running.
    from mcp_server.store import Store

    store = Store("postgresql://nobody@localhost:1/none")
    assert store.connection_string.startswith("postgresql://")
