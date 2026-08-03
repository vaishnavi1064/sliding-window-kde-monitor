"""Fetch the MetroPT-3 dataset and convert it to Parquet for fast replay.

MetroPT-3 (UCI dataset 791): Air Production Unit sensors from a Metro do Porto
train, 1,516,948 readings every 10s (0.1 Hz), Feb-Sep 2020, 15 sensor channels. Roughly
208 MB as CSV. The data is NOT committed to this repo.

The CSV is converted to Parquet because the producer re-reads the whole stream
on every run, and Parquet is dramatically faster to load and much smaller.

Usage:  python -m scripts.download_data
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
PARQUET_PATH = DATA_DIR / "metropt3.parquet"


def download(force: bool = False) -> Path:
    import requests

    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / "MetroPT3(AirCompressor).csv"
    if csv_path.exists() and not force:
        print(f"CSV already present at {csv_path}")
        return csv_path

    print(f"Downloading {URL}")
    response = requests.get(URL, stream=True, timeout=300)
    response.raise_for_status()

    payload = io.BytesIO()
    downloaded = 0
    next_report = 25_000_000
    for chunk in response.iter_content(chunk_size=1 << 20):
        payload.write(chunk)
        downloaded += len(chunk)
        # Report every 25 MB rather than every chunk: carriage-return progress
        # produces megabytes of noise when stdout is a file rather than a tty.
        if downloaded >= next_report:
            print(f"  {downloaded / 1e6:.0f} MB", flush=True)
            next_report += 25_000_000
    print(f"  {downloaded / 1e6:.0f} MB total")

    payload.seek(0)
    with zipfile.ZipFile(payload) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV inside the archive; contents: {archive.namelist()}")
        extracted = archive.extract(names[0], DATA_DIR)
        extracted_path = Path(extracted)
        if extracted_path != csv_path:
            extracted_path.replace(csv_path)

    print(f"Extracted to {csv_path}")
    return csv_path


CHUNK_ROWS = 200_000


def to_parquet(csv_path: Path, force: bool = False) -> Path:
    """Convert the CSV to Parquet in bounded memory.

    Read whole, this file expands to several GB -- pandas widens every column to
    float64 and the parse holds the source alongside the frame. Streaming it in
    chunks with narrow dtypes keeps the peak in the hundreds of MB, which also
    means the conversion runs happily alongside the Docker stack.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from streaming.features import ANALOG_COLUMNS, DIGITAL_COLUMNS

    if PARQUET_PATH.exists() and not force:
        print(f"Parquet already present at {PARQUET_PATH}")
        return PARQUET_PATH

    # The leading unnamed column is a written-out pandas index; skip it rather
    # than reading and dropping it. Digital channels are stored as "1.0"/"0.0",
    # so they parse as float and get narrowed afterwards.
    columns = ["timestamp", *ANALOG_COLUMNS, *DIGITAL_COLUMNS]
    dtypes = {column: "float32" for column in (*ANALOG_COLUMNS, *DIGITAL_COLUMNS)}

    print(f"Converting to Parquet in {CHUNK_ROWS:,}-row chunks ...")
    writer: pq.ParquetWriter | None = None
    rows = 0
    first_timestamp = last_timestamp = None

    try:
        reader = pd.read_csv(
            csv_path,
            usecols=columns,
            dtype=dtypes,
            parse_dates=["timestamp"],
            chunksize=CHUNK_ROWS,
        )
        for chunk in reader:
            chunk = chunk[columns]
            for column in DIGITAL_COLUMNS:
                chunk[column] = chunk[column].astype("int8")

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(PARQUET_PATH, table.schema, compression="zstd")
                first_timestamp = chunk["timestamp"].iloc[0]
            writer.write_table(table)

            rows += len(chunk)
            last_timestamp = chunk["timestamp"].iloc[-1]
            print(f"  {rows:,} rows", flush=True)
    finally:
        if writer is not None:
            writer.close()

    csv_mb = csv_path.stat().st_size / 1e6
    parquet_mb = PARQUET_PATH.stat().st_size / 1e6
    print(f"\nWrote {PARQUET_PATH}")
    print(f"  rows  : {rows:,}")
    print(f"  span  : {first_timestamp} -> {last_timestamp}")
    print(f"  size  : {csv_mb:.0f} MB CSV -> {parquet_mb:.0f} MB Parquet")
    return PARQUET_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download and reconvert")
    args = parser.parse_args()

    csv_path = download(force=args.force)
    to_parquet(csv_path, force=args.force)

    print(
        "\nGround truth: MetroPT-3 is unlabeled. The four documented air-leak\n"
        "failures come from a separate maintenance report and are encoded in\n"
        "streaming/failures.py -- labels are derived by joining on timestamp."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
