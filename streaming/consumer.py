"""Consume the sensor stream, maintain the SW-AKDE sketch, expose Prometheus metrics.

This process owns the **logical clock**: a monotonic counter incremented once per
accepted reading, which is what drives sketch window expiry. MetroPT timestamps
are real wall-clock values that repeat, skip and occasionally go backwards, and
feeding those to the exponential histograms would corrupt expiry silently
(CLAUDE.md Finding B). Timestamp irregularities are still *reported* as data
quality issues -- they just don't drive the clock.

Usage:  python -m streaming.consumer
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from streaming.config import SETTINGS
from streaming.features import ANALOG_COLUMNS, WarmupStandardizer, extract
from streaming.quality import QualityMonitor
from streaming.scoring import RollingAnomalyScorer
from streaming.sketch_factory import build_sketch

RECORDS = Counter("swakde_records_total", "Records consumed", ["outcome"])
QUALITY_ISSUES = Counter("swakde_quality_issues_total", "Data quality issues", ["kind"])
UPDATE_SECONDS = Histogram(
    "swakde_update_seconds",
    "Sketch update latency",
    buckets=(0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05),
)
QUERY_SECONDS = Histogram(
    "swakde_query_seconds",
    "Sketch query latency",
    buckets=(0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05),
)
DENSITY = Gauge("swakde_density", "Sliding-window kernel density at the latest reading")
DENSITY_SMOOTHED = Gauge(
    "swakde_density_smoothed",
    "EWMA-smoothed density -- the series the anomaly score is actually computed on",
)
ANOMALY_SCORE = Gauge("swakde_anomaly_score", "Sigmas below the rolling density mean")
LOGICAL_CLOCK = Gauge("swakde_logical_clock", "Monotonic per-event counter driving expiry")
CELLS = Gauge("swakde_cells", "Populated sketch cells (memory proxy)")
WARMED_UP = Gauge("swakde_warmed_up", "1 once the feature standardizer is fitted")
WINDOW_FULL = Gauge(
    "swakde_window_full",
    "1 once the sliding window holds a full complement and density is comparable over time",
)
INJECTED = Gauge("swakde_injected_fault", "1 while the producer is injecting a synthetic fault")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", default=SETTINGS.bootstrap_servers)
    parser.add_argument("--topic", default=SETTINGS.topic)
    parser.add_argument("--group", default=SETTINGS.consumer_group)
    parser.add_argument("--kernel", choices=("euclidean", "angular"), default="euclidean")
    parser.add_argument("--metrics-port", type=int, default=SETTINGS.metrics_port)
    parser.add_argument("--from-beginning", action="store_true", default=True)
    return parser


def parse_timestamp(record: dict) -> datetime | None:
    raw = record.get("timestamp")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def main() -> int:
    from confluent_kafka import Consumer, KafkaError

    args = build_parser().parse_args()

    start_http_server(args.metrics_port)
    print(f"Metrics on :{args.metrics_port}/metrics")

    sketch = build_sketch(kernel=args.kernel, dim=len(ANALOG_COLUMNS))
    # Report the resolved core, not the requested one: with SKETCH_BACKEND=auto a
    # container built without a compiler silently falls back to the Python core,
    # and that should be visible in the logs rather than inferred from throughput.
    print(f"Sketch core: {sketch.backend}")
    standardizer = WarmupStandardizer(len(ANALOG_COLUMNS), warmup=SETTINGS.warmup)
    quality = QualityMonitor(ANALOG_COLUMNS)
    scorer = RollingAnomalyScorer()

    consumer = Consumer(
        {
            "bootstrap.servers": args.bootstrap,
            "group.id": args.group,
            "auto.offset.reset": "earliest" if args.from_beginning else "latest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([args.topic])
    print(f"Consuming {args.topic} from {args.bootstrap} (kernel={args.kernel})")

    # The logical clock. Increments once per reading actually fed to the sketch,
    # so window semantics stay exact regardless of wall-clock irregularities.
    clock = 0
    reported = time.perf_counter()
    last_error = ""

    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                error = message.error()
                # Reaching the end of a partition, and the topic not existing
                # yet, are both normal while waiting for the producer to start.
                if error.code() == KafkaError._PARTITION_EOF:
                    continue
                RECORDS.labels(outcome="error").inc()
                if str(error) != last_error:
                    last_error = str(error)
                    print(f"  kafka: {error}")
                continue

            try:
                record = json.loads(message.value())
            except (ValueError, TypeError):
                RECORDS.labels(outcome="undecodable").inc()
                continue

            values = extract(record)
            when = parse_timestamp(record)
            report = quality.check(record, values, when)

            for _ in report.missing_fields:
                QUALITY_ISSUES.labels(kind="missing_field").inc()
            for _ in report.out_of_range:
                QUALITY_ISSUES.labels(kind="out_of_range").inc()
            if report.timestamp_regression:
                QUALITY_ISSUES.labels(kind="timestamp_regression").inc()
            if report.duplicate_timestamp:
                QUALITY_ISSUES.labels(kind="duplicate_timestamp").inc()
            if report.gap_seconds:
                QUALITY_ISSUES.labels(kind="gap").inc()

            if not report.usable:
                RECORDS.labels(outcome="dropped").inc()
                continue

            INJECTED.set(1 if record.get("_injected") else 0)

            standardizer.observe(values)
            if not standardizer.fitted:
                RECORDS.labels(outcome="warmup").inc()
                continue
            WARMED_UP.set(1)

            x = standardizer.transform(values)
            if not np.all(np.isfinite(x)):
                RECORDS.labels(outcome="dropped").inc()
                continue

            clock += 1
            started = time.perf_counter()
            sketch.update(x, clock)
            UPDATE_SECONDS.observe(time.perf_counter() - started)
            RECORDS.labels(outcome="accepted").inc()
            LOGICAL_CLOCK.set(clock)

            # Querying every reading would roughly double the work for no gain
            # at these replay rates.
            if clock % SETTINGS.query_every == 0:
                started = time.perf_counter()
                density = sketch.query(x, clock)
                QUERY_SECONDS.observe(time.perf_counter() - started)
                DENSITY.set(density)
                CELLS.set(sketch.cell_count)

                # Density is not comparable across time until the sliding
                # window is full: while it fills, density ramps up from zero
                # purely as an artefact of how much data is in the window.
                # Feeding that ramp to the scorer poisons its baseline -- the
                # spread it measures is dominated by the ramp rather than by
                # real variation, which flattens the score during an actual
                # fault. Only score once the window holds a full complement.
                if clock >= sketch.window_size:
                    WINDOW_FULL.set(1)
                    ANOMALY_SCORE.set(scorer.score(density))
                    if scorer.smoothed_density is not None:
                        DENSITY_SMOOTHED.set(scorer.smoothed_density)

            now = time.perf_counter()
            if now - reported >= 10.0:
                print(
                    f"  clock={clock:,}  density={DENSITY._value.get():.2f}  "
                    f"score={ANOMALY_SCORE._value.get():.2f}  cells={sketch.cell_count:,}"
                )
                reported = now
    except KeyboardInterrupt:
        print("\nStopping")
    finally:
        consumer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
