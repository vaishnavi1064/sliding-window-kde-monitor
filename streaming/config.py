"""Environment-driven configuration shared by producer and consumer."""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    bootstrap_servers: str = _env("KAFKA_BOOTSTRAP", "localhost:9092")
    topic: str = _env("KAFKA_TOPIC", "metropt.sensors")
    consumer_group: str = _env("KAFKA_GROUP", "swakde-monitor")

    parquet_path: Path = Path(_env("METROPT_PARQUET", str(REPO_ROOT / "data" / "metropt3.parquet")))

    # Replay speed. MetroPT-3 samples every 10s over ~7 months, so real-time
    # replay would run for ~176 days; the default compresses that to ~25 minutes.
    speedup: float = float(_env("REPLAY_SPEEDUP", "1000"))
    max_records: int = int(_env("REPLAY_MAX_RECORDS", "0"))  # 0 = whole file

    # Sketch parameters. rows=400 sits at ~2,000 updates/s in the Python core
    # (docs/PERFORMANCE.md), which is far above the replay rate.
    rows: int = int(_env("SKETCH_ROWS", "400"))
    k: int = int(_env("SKETCH_K", "3"))
    window_size: int = int(_env("SKETCH_WINDOW", "3600"))  # ~10 hours at 0.1 Hz
    eh_relative_error: float = float(_env("SKETCH_EH_ERROR", "0.1"))
    lsh_width: float = float(_env("SKETCH_WIDTH", "2.0"))  # p-stable bucket width

    warmup: int = int(_env("SKETCH_WARMUP", "2000"))
    # Density is only meaningful once the window has filled.
    query_every: int = int(_env("QUERY_EVERY", "10"))

    metrics_port: int = int(_env("METRICS_PORT", "8000"))


SETTINGS = Settings()
