"""End-to-end check of the MCP server against a live Postgres.

Calls the three tools the way an agent would and prints what comes back, so the
Phase 4 definition of done ("an MCP client/agent queries health + recent
anomalies end-to-end") is demonstrable rather than asserted.

Requires: docker compose up -d postgres, then python -m scripts.load_alerts

Usage:  python -m scripts.verify_mcp
"""

import json
import sys

from mcp_server.server import explain_alert, get_asset_health, list_recent_anomalies
from mcp_server.store import ASSET_ID, Store


def _call(tool, **kwargs):
    """Invoke a registered MCP tool's underlying function."""
    fn = getattr(tool, "fn", None) or getattr(tool, "__wrapped__", None) or tool
    return fn(**kwargs)


def show(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, default=str))


def main() -> int:
    store = Store()
    try:
        health = store.asset_health(ASSET_ID)
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot reach Postgres at {store.connection_string}\n  {exc}")
        print("\nStart it and load data first:")
        print("  docker compose up -d postgres")
        print("  python -m scripts.load_alerts")
        return 1

    if health is None:
        print(f"No asset '{ASSET_ID}' in the database. Run: python -m scripts.load_alerts")
        return 1

    show("get_asset_health", _call(get_asset_health))

    listing = _call(list_recent_anomalies, limit=3)
    show("list_recent_anomalies(limit=3)", listing)

    confirmed = _call(list_recent_anomalies, limit=5, only_confirmed=True)
    show("list_recent_anomalies(only_confirmed=True)", confirmed)

    # Explain a confirmed alert if there is one, else the most recent.
    candidates = confirmed["anomalies"] or listing["anomalies"]
    if candidates:
        show(f"explain_alert({candidates[0]['anomaly_id']})",
             _call(explain_alert, anomaly_id=candidates[0]["anomaly_id"]))

    show("explain_alert(999999)  [unknown id]", _call(explain_alert, anomaly_id=999999))

    print("\n--- checks ---")
    checks = [
        ("asset health returned a status", bool(_call(get_asset_health).get("status"))),
        ("alert history is non-empty", len(listing["anomalies"]) > 0),
        ("at least one alert matched a documented failure",
         len(confirmed["anomalies"]) > 0),
        ("every response carries the interpretation caveat",
         "at onset" in _call(get_asset_health)["interpretation"]),
        ("unknown alert id returns an error rather than raising",
         "error" in _call(explain_alert, anomaly_id=999999)),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
