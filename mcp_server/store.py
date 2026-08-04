"""Postgres access for the MCP server.

Kept separate from the tool definitions so the query logic is testable without
an MCP client, and so the tools stay thin.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

SCHEMA = Path(__file__).resolve().parent / "schema.sql"

ASSET_ID = "metropt-apu"


def dsn() -> str:
    return os.environ.get(
        "POSTGRES_DSN", "postgresql://swakde:swakde@localhost:5432/swakde"
    )


@dataclass(frozen=True)
class Anomaly:
    anomaly_id: int
    asset_id: str
    started_at: datetime
    ended_at: datetime
    peak_score: float
    threshold: float
    method: str
    matched_failure: str | None
    lead_hours: float | None

    @property
    def duration(self) -> timedelta:
        return self.ended_at - self.started_at

    @property
    def is_false_alarm(self) -> bool:
        return self.matched_failure is None


@dataclass(frozen=True)
class AssetHealth:
    asset_id: str
    description: str
    anomalies_24h: int
    anomalies_7d: int
    latest: Anomaly | None
    total_anomalies: int
    matched_failures: int
    status: str
    window_end: datetime


def health_status(anomalies_24h: int, latest: Anomaly | None) -> str:
    """Map recent alert activity onto a coarse state.

    Deliberately coarse. The detector's own evaluation says it identifies
    failures at onset rather than predicting them, so a "healthy" verdict here
    means "nothing detected recently", not "nothing developing".
    """
    if anomalies_24h == 0:
        return "healthy"
    if latest is not None and latest.peak_score >= 2.0:
        return "critical"
    return "warning"


class Store:
    """Thin synchronous wrapper. Opens a connection per call, which is ample for
    an MCP server answering occasional agent queries."""

    def __init__(self, connection_string: str | None = None):
        self.connection_string = connection_string or dsn()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.connection_string)

    def initialise(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(SCHEMA.read_text(encoding="utf-8"))
            conn.commit()

    def upsert_asset(
        self, asset_id: str, description: str, sensor_count: int, sampling_hz: float
    ) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO assets (asset_id, description, sensor_count, sampling_hz)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (asset_id) DO UPDATE SET
                    description = EXCLUDED.description,
                    sensor_count = EXCLUDED.sensor_count,
                    sampling_hz = EXCLUDED.sampling_hz
                """,
                (asset_id, description, sensor_count, sampling_hz),
            )
            conn.commit()

    def replace_failures(self, asset_id: str, failures: list[dict]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM documented_failures WHERE asset_id = %s", (asset_id,))
            for failure in failures:
                cur.execute(
                    """
                    INSERT INTO documented_failures
                        (failure_id, asset_id, started_at, ended_at, kind)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (failure["failure_id"], asset_id, failure["started_at"],
                     failure["ended_at"], failure["kind"]),
                )
            conn.commit()

    def replace_anomalies(self, asset_id: str, method: str, rows: list[dict]) -> int:
        """Replace this asset+method's episodes. Idempotent, so the loader can
        be re-run without accumulating duplicates."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM anomalies WHERE asset_id = %s AND method = %s",
                (asset_id, method),
            )
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO anomalies (asset_id, started_at, ended_at, peak_score,
                                           threshold, method, matched_failure, lead_hours)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (asset_id, row["started_at"], row["ended_at"], row["peak_score"],
                     row["threshold"], method, row.get("matched_failure"),
                     row.get("lead_hours")),
                )
            conn.commit()
            return len(rows)

    def _row_to_anomaly(self, row) -> Anomaly:
        return Anomaly(*row)

    def recent_anomalies(
        self, asset_id: str = ASSET_ID, limit: int = 20,
        since: datetime | None = None, only_matched: bool = False,
    ) -> list[Anomaly]:
        clauses = ["asset_id = %s"]
        params: list = [asset_id]
        if since is not None:
            clauses.append("started_at >= %s")
            params.append(since)
        if only_matched:
            clauses.append("matched_failure IS NOT NULL")
        params.append(limit)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT anomaly_id, asset_id, started_at, ended_at, peak_score,
                       threshold, method, matched_failure, lead_hours
                FROM anomalies WHERE {' AND '.join(clauses)}
                ORDER BY started_at DESC LIMIT %s
                """,
                params,
            )
            return [self._row_to_anomaly(r) for r in cur.fetchall()]

    def get_anomaly(self, anomaly_id: int) -> Anomaly | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT anomaly_id, asset_id, started_at, ended_at, peak_score,
                       threshold, method, matched_failure, lead_hours
                FROM anomalies WHERE anomaly_id = %s
                """,
                (anomaly_id,),
            )
            row = cur.fetchone()
            return self._row_to_anomaly(row) if row else None

    def asset_health(self, asset_id: str = ASSET_ID) -> AssetHealth | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT description FROM assets WHERE asset_id = %s", (asset_id,)
            )
            row = cur.fetchone()
            if row is None:
                return None
            description = row[0]

            # "Recent" is relative to the newest record, not to wall clock: this
            # is replayed 2020 data, so anchoring on now() would always report
            # zero activity.
            cur.execute(
                "SELECT MAX(ended_at) FROM anomalies WHERE asset_id = %s", (asset_id,)
            )
            window_end = cur.fetchone()[0] or datetime(1970, 1, 1)

            cur.execute(
                """
                SELECT
                  COUNT(*) FILTER (WHERE started_at >= %s),
                  COUNT(*) FILTER (WHERE started_at >= %s),
                  COUNT(*),
                  COUNT(*) FILTER (WHERE matched_failure IS NOT NULL)
                FROM anomalies WHERE asset_id = %s
                """,
                (window_end - timedelta(hours=24), window_end - timedelta(days=7),
                 asset_id),
            )
            day, week, total, matched = cur.fetchone()

        latest = next(iter(self.recent_anomalies(asset_id, limit=1)), None)
        return AssetHealth(
            asset_id=asset_id,
            description=description,
            anomalies_24h=day,
            anomalies_7d=week,
            latest=latest,
            total_anomalies=total,
            matched_failures=matched,
            status=health_status(day, latest),
            window_end=window_end,
        )

    def documented_failures(self, asset_id: str = ASSET_ID) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT failure_id, started_at, ended_at, kind
                FROM documented_failures WHERE asset_id = %s ORDER BY started_at
                """,
                (asset_id,),
            )
            return [
                {"failure_id": f, "started_at": s, "ended_at": e, "kind": k}
                for f, s, e, k in cur.fetchall()
            ]
