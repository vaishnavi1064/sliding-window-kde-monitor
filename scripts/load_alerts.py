"""Load detected anomaly episodes into Postgres for the MCP server.

Turns a cached score series into the durable alert history the MCP tools answer
questions about. Episodes come from the same logic the evaluation uses, so what
an agent sees matches what the report says -- rather than the server having its
own private notion of an alert.

Usage:
    python -m scripts.load_alerts                       # 5% operating point
    python -m scripts.load_alerts --alerting 0.02
"""

import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

from evaluation.labels import labelled_windows
from evaluation.metrics import _alert_episodes
from mcp_server.store import ASSET_ID, Store
from scripts.run_evaluation import cache_path, rescore
from streaming.config import SETTINGS
from streaming.failures import FAILURES
from streaming.features import ANALOG_COLUMNS


def episodes_for(series: pd.DataFrame, alerting: float) -> tuple[list[dict], float]:
    scores = rescore(series)
    threshold = float(np.quantile(scores, 1.0 - alerting))
    times = pd.to_datetime(series["timestamp"]).to_numpy()
    alerting_mask = scores > threshold

    windows = labelled_windows()
    rows = []
    for start, end in _alert_episodes(times, alerting_mask, timedelta(hours=1)):
        span = (times >= np.datetime64(start)) & (times <= np.datetime64(end))
        matched = next((w for w in windows if w.contains(start)), None)
        rows.append({
            "started_at": start,
            "ended_at": end,
            "peak_score": float(scores[span].max()) if span.any() else float(threshold),
            "threshold": threshold,
            "matched_failure": matched.name if matched else None,
            "lead_hours": (
                matched.lead_time(start).total_seconds() / 3600.0 if matched else None
            ),
        })
    return rows, threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="swakde")
    parser.add_argument("--alerting", type=float, default=0.05,
                        help="alarm budget as a fraction of time alerting")
    args = parser.parse_args()

    path = cache_path(args.method)
    if not path.exists():
        raise SystemExit(f"no cached series at {path.name}; run make evaluate first")

    store = Store()
    store.initialise()
    store.upsert_asset(
        ASSET_ID,
        "Metro do Porto train, Air Production Unit (MetroPT-3)",
        sensor_count=15,
        sampling_hz=0.1,
    )
    store.replace_failures(
        ASSET_ID,
        [
            {"failure_id": f.name, "started_at": f.start, "ended_at": f.end,
             "kind": f.kind}
            for f in FAILURES
        ],
    )

    rows, threshold = episodes_for(pd.read_parquet(path), args.alerting)
    written = store.replace_anomalies(ASSET_ID, args.method, rows)

    matched = sum(1 for r in rows if r["matched_failure"])
    print(f"Loaded {written} alert episodes for {ASSET_ID} [{args.method}]")
    print(f"  threshold      {threshold:.3f} (top {args.alerting * 100:.0f}% of scores)")
    print(f"  matched a documented failure: {matched}")
    print(f"  not matched (false alarms)  : {written - matched}")
    print(f"  documented failures loaded  : {len(FAILURES)}")
    print(f"  analog channels             : {len(ANALOG_COLUMNS)} of "
          f"{SETTINGS.rows}-row sketch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
