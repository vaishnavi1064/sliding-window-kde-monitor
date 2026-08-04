"""MCP server exposing the monitor to AI agents.

Three tools, per the project brief:

    get_asset_health        current state of the asset
    list_recent_anomalies   the alert history
    explain_alert           what one alert was, and whether it was real

Design notes worth stating, because they are the difference between a useful
agent interface and a misleading one:

* **Tools return the detector's own honest accounting.** `matched_failure` is
  populated from the documented maintenance report, so `explain_alert` can say
  "this coincided with a real air leak" or "this did not" rather than only
  quoting a score. An agent that can only see scores cannot tell a user whether
  to trust them.

* **Every response carries the caveats the evaluation established.** The
  detector identifies failures at onset rather than predicting them, and two of
  the four documented failures have almost no signature in these channels. An
  agent relaying "healthy" needs to know that means "nothing detected recently",
  not "nothing developing". Encoding that in the tool output is the only way it
  reaches the end user.

Run with:  python -m mcp_server.server
"""

from datetime import timedelta

from mcp.server.mcpserver import MCPServer

from mcp_server.store import ASSET_ID, Anomaly, Store

mcp = MCPServer(
    name="swakde-monitor",
    instructions=(
        "Sliding-window kernel-density anomaly monitoring for an industrial "
        "air production unit. Detects failures at onset rather than predicting "
        "them; read the interpretation field on every response before relaying "
        "a health verdict to a user."
    ),
)
# Constructed on first use, not at import. Resolving credentials eagerly would
# make `import mcp_server.server` fail wherever they are absent -- including in
# unit tests, which exercise the tool logic without a database.
_store_instance: Store | None = None


def _get_store() -> Store:
    global _store_instance
    if _store_instance is None:
        _store_instance = Store()
    return _store_instance

# Stated on every health response. These are conclusions from the Phase 3
# evaluation (docs/EVALUATION.md), not hedging boilerplate.
DETECTOR_CAVEATS = (
    "This detector identifies failures at onset rather than predicting them: "
    "measured lead time against the four documented MetroPT-3 failures is "
    "approximately zero. 'healthy' therefore means nothing has been detected "
    "recently, not that nothing is developing. Two of those four failures also "
    "have almost no signature in these sensor channels beforehand."
)


def _describe(anomaly: Anomaly) -> dict:
    return {
        "anomaly_id": anomaly.anomaly_id,
        "started_at": anomaly.started_at.isoformat(),
        "ended_at": anomaly.ended_at.isoformat(),
        "duration_hours": round(anomaly.duration.total_seconds() / 3600.0, 2),
        "peak_score": round(anomaly.peak_score, 3),
        "threshold": round(anomaly.threshold, 3),
        "method": anomaly.method,
        "coincided_with_documented_failure": anomaly.matched_failure,
        "lead_hours": (
            round(anomaly.lead_hours, 2) if anomaly.lead_hours is not None else None
        ),
    }


def alert_verdict(anomaly: Anomaly) -> str:
    """Plain-language judgement on whether an alert was real.

    Pure so it can be tested without a database, and so the wording an agent
    relays is pinned by tests rather than drifting.
    """
    if not anomaly.matched_failure:
        return (
            "This alert did not coincide with any documented failure, so on the "
            "evaluation's accounting it is a false alarm. The measured rate is "
            "roughly 0.35 such alarms per day of normal operation at this "
            "threshold."
        )

    lead = anomaly.lead_hours or 0.0
    timing = (
        f"{lead:.1f}h before that failure's recorded start"
        if lead > 0.5
        else "essentially at the failure's recorded start"
    )
    return (
        f"This alert coincided with documented failure "
        f"'{anomaly.matched_failure}' ({anomaly.method}), firing {timing}."
    )


def why_it_fired(anomaly: Anomaly) -> str:
    multiple = anomaly.peak_score / anomaly.threshold if anomaly.threshold else 0.0
    return (
        f"Kernel density around the current sensor reading fell to "
        f"{multiple:.1f}x the alerting threshold, measured as robust deviations "
        f"below the rolling median of log density. A drop means recent readings "
        f"moved into a region of the feature space that had gone sparse within "
        f"the sliding window."
    )


@mcp.tool()
def get_asset_health(asset_id: str = ASSET_ID) -> dict:
    """Current health of a monitored asset.

    Returns a coarse status plus recent alert counts. Status is derived from
    detected alert episodes only.
    """
    health = _get_store().asset_health(asset_id)
    if health is None:
        return {"error": f"unknown asset '{asset_id}'"}

    return {
        "asset_id": health.asset_id,
        "description": health.description,
        "status": health.status,
        "as_of": health.window_end.isoformat(),
        "anomalies_last_24h": health.anomalies_24h,
        "anomalies_last_7d": health.anomalies_7d,
        "anomalies_total": health.total_anomalies,
        "anomalies_matching_documented_failures": health.matched_failures,
        "latest_anomaly": _describe(health.latest) if health.latest else None,
        "interpretation": DETECTOR_CAVEATS,
    }


@mcp.tool()
def list_recent_anomalies(
    asset_id: str = ASSET_ID,
    limit: int = 10,
    hours: int | None = None,
    only_confirmed: bool = False,
) -> dict:
    """List detected anomaly episodes, newest first.

    One entry per episode, not per reading above threshold. Set
    `only_confirmed` to return just those coinciding with a documented failure.
    `hours` limits to the period before the most recent record, since this is
    replayed historical data rather than a live feed.
    """
    health = _get_store().asset_health(asset_id)
    if health is None:
        return {"error": f"unknown asset '{asset_id}'"}

    since = health.window_end - timedelta(hours=hours) if hours else None
    anomalies = _get_store().recent_anomalies(
        asset_id, limit=limit, since=since, only_matched=only_confirmed
    )
    return {
        "asset_id": asset_id,
        "count": len(anomalies),
        "as_of": health.window_end.isoformat(),
        "filtered_to_confirmed": only_confirmed,
        "anomalies": [_describe(a) for a in anomalies],
        "note": (
            "An episode without a coincided_with_documented_failure value is a "
            "false alarm on the evaluation's own accounting."
        ),
    }


@mcp.tool()
def explain_alert(anomaly_id: int) -> dict:
    """Explain one alert: what fired, how strongly, and whether it was real.

    The judgement an agent actually needs is not the score but whether the alert
    coincided with a documented failure, so that is stated plainly.
    """
    anomaly = _get_store().get_anomaly(anomaly_id)
    if anomaly is None:
        return {"error": f"no anomaly with id {anomaly_id}"}

    return {
        **_describe(anomaly),
        "verdict": alert_verdict(anomaly),
        "why_it_fired": why_it_fired(anomaly),
        "interpretation": DETECTOR_CAVEATS,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
