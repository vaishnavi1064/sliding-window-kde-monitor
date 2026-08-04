-- Alert history and asset metadata for the MCP server.
--
-- Deliberately small. The sketch itself is not stored here: the MCP server
-- answers questions about *what was detected*, which is a durable record, while
-- the sketch is live in-memory state owned by the consumer.

CREATE TABLE IF NOT EXISTS assets (
    asset_id      TEXT PRIMARY KEY,
    description   TEXT NOT NULL,
    sensor_count  INTEGER NOT NULL,
    sampling_hz   DOUBLE PRECISION NOT NULL
);

-- One row per alert episode, not per sample above threshold. A sustained fault
-- produces thousands of consecutive breaches; an operator sees one alert, and
-- storing them per sample would make any rate meaningless.
CREATE TABLE IF NOT EXISTS anomalies (
    anomaly_id    BIGSERIAL PRIMARY KEY,
    asset_id      TEXT NOT NULL REFERENCES assets(asset_id),
    started_at    TIMESTAMP NOT NULL,
    ended_at      TIMESTAMP NOT NULL,
    peak_score    DOUBLE PRECISION NOT NULL,
    threshold     DOUBLE PRECISION NOT NULL,
    method        TEXT NOT NULL,
    -- Which documented failure this episode falls inside, if any. NULL means it
    -- did not coincide with a known failure -- i.e. a false alarm on the
    -- evaluation's own accounting.
    matched_failure TEXT,
    lead_hours    DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS anomalies_started_at_idx ON anomalies (started_at DESC);
CREATE INDEX IF NOT EXISTS anomalies_asset_idx ON anomalies (asset_id, started_at DESC);

-- The ground-truth maintenance report, so explain_alert can say whether an
-- episode lines up with a real failure rather than only reporting a score.
CREATE TABLE IF NOT EXISTS documented_failures (
    failure_id    TEXT PRIMARY KEY,
    asset_id      TEXT NOT NULL REFERENCES assets(asset_id),
    started_at    TIMESTAMP NOT NULL,
    ended_at      TIMESTAMP NOT NULL,
    kind          TEXT NOT NULL
);
