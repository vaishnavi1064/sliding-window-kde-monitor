"""Replay MetroPT-3 into Kafka as a live stream.

MetroPT-3 samples every 10s over roughly seven months, so real-time replay would
run for ~176 days. Replay is therefore accelerated by a configurable factor.

Also supports injecting a synthetic fault, which is how the Phase 2 alerting
path is demonstrated end to end without waiting for one of the four real
failures to come round in the replay.

Usage:
    python -m streaming.producer
    python -m streaming.producer --speedup 5000 --max-records 200000
    python -m streaming.producer --inject-anomaly-at 50000 --inject-duration 3000
"""

import argparse
import json
import time

import pandas as pd

from streaming.config import SETTINGS
from streaming.features import ANALOG_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speedup", type=float, default=SETTINGS.speedup)
    parser.add_argument("--max-records", type=int, default=SETTINGS.max_records)
    parser.add_argument("--topic", default=SETTINGS.topic)
    parser.add_argument("--bootstrap", default=SETTINGS.bootstrap_servers)
    parser.add_argument(
        "--inject-anomaly-at",
        type=int,
        default=0,
        help="record index at which to start injecting a synthetic fault (0 = never)",
    )
    parser.add_argument("--inject-duration", type=int, default=3000)
    parser.add_argument(
        "--inject-scale",
        type=float,
        default=3.0,
        help="multiplier applied to analog channels during the injected fault",
    )
    return parser


def main() -> int:
    from confluent_kafka import Producer

    args = build_parser().parse_args()

    if not SETTINGS.parquet_path.exists():
        raise SystemExit(
            f"{SETTINGS.parquet_path} not found. Run: python -m scripts.download_data"
        )

    print(f"Loading {SETTINGS.parquet_path} ...")
    frame = pd.read_parquet(SETTINGS.parquet_path)
    if args.max_records:
        frame = frame.iloc[: args.max_records]
    print(f"  {len(frame):,} records")

    producer = Producer(
        {
            "bootstrap.servers": args.bootstrap,
            "linger.ms": 50,
            "compression.type": "lz4",
            "queue.buffering.max.messages": 1_000_000,
        }
    )

    inject_start = args.inject_anomaly_at
    inject_end = inject_start + args.inject_duration if inject_start else 0
    if inject_start:
        print(f"Injecting synthetic fault over records {inject_start:,}-{inject_end:,}")

    # Wall-clock pacing. Note this is only about replay realism -- the sketch's
    # own clock is a monotonic counter maintained by the consumer (Finding B).
    interval = 1.0 / args.speedup
    started = time.perf_counter()
    sent = 0

    records = frame.to_dict("records")
    for index, record in enumerate(records):
        if "timestamp" in record and hasattr(record["timestamp"], "isoformat"):
            record["timestamp"] = record["timestamp"].isoformat()

        injected = inject_start and inject_start <= index < inject_end
        if injected:
            for column in ANALOG_COLUMNS:
                if column in record:
                    record[column] = float(record[column]) * args.inject_scale
        record["_injected"] = bool(injected)

        producer.produce(args.topic, json.dumps(record).encode())
        sent += 1

        if sent % 10_000 == 0:
            producer.poll(0)
            elapsed = time.perf_counter() - started
            print(f"\r  sent {sent:,}  ({sent / elapsed:,.0f} rec/s)", end="", flush=True)

        target = started + (index + 1) * interval
        remaining = target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    producer.flush(30)
    elapsed = time.perf_counter() - started
    print(f"\nSent {sent:,} records in {elapsed:.1f}s ({sent / elapsed:,.0f} rec/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
