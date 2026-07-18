"""Convert raw OPS-SAT CSV files into our standard parquet layout.

Consumes the extracted output of `scripts/download_ops_sat.py` and
produces a directory that the existing `parquet_telemetry` loader
consumes directly:

    <output>/
      data.parquet        timestamp_ms + one column per channel (float32)
      labels.parquet      timestamp_ms + is_anomaly (bool)
      channels.yaml       per-channel metadata (name, units, subsystem, rate)
      manifest.yaml       dataset name, subsystem, size, anomaly rate, source

Schema-adaptive: the CSV column-naming conventions in OPS-SAT releases
vary between the 2020 Case Challenge and the 2023 OPS-SAT-AD variant.
`_read_ops_sat_raw()` tries a small set of common column names and
fails loudly if none match — at which point you edit that function to
match the real file layout.

Usage:
    python scripts/convert_ops_sat_to_parquet.py \\
        --raw data/ops_sat_raw \\
        --output data/ops_sat

    # Downsample from 1 Hz to 0.1 Hz to save disk:
    python scripts/convert_ops_sat_to_parquet.py --downsample 10

    # Only convert a subset of channels:
    python scripts/convert_ops_sat_to_parquet.py --channels channel_45,channel_46
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


# ── Schema variants the converter knows how to handle ─────────────────
#
# If your OPS-SAT release uses different column names, add a variant
# tuple below or edit _read_ops_sat_raw() directly.

_TELEMETRY_FILENAMES = ("telemetry.csv", "hk_telemetry.csv", "ops_sat_telemetry.csv")
_ANOMALY_FILENAMES = ("anomalies.csv", "labels.csv", "ops_sat_anomalies.csv")
_TIMESTAMP_COLUMNS = ("timestamp", "time", "utc", "ts", "datetime")
_ANOMALY_START_COLUMNS = ("start_time", "start", "t_start", "begin")
_ANOMALY_END_COLUMNS = ("end_time", "end", "t_end", "finish")


def _find_file(raw: Path, candidates: tuple[str, ...]) -> Path:
    for cand in candidates:
        for p in raw.rglob(cand):
            return p
    raise FileNotFoundError(
        f"No OPS-SAT file matching any of {candidates} under {raw}. "
        f"If the release uses different filenames, add them to _TELEMETRY_FILENAMES / "
        f"_ANOMALY_FILENAMES in scripts/convert_ops_sat_to_parquet.py."
    )


def _read_ops_sat_raw(raw: Path):
    """Load raw OPS-SAT files into numpy arrays. Returns (timestamps_ms, telemetry_arr, channel_names, anomaly_ranges)."""
    import pandas as pd

    tel_path = _find_file(raw, _TELEMETRY_FILENAMES)
    print(f"  telemetry: {tel_path}")
    tel = pd.read_csv(tel_path)

    # Timestamp column
    ts_col = next((c for c in _TIMESTAMP_COLUMNS if c in tel.columns), None)
    if ts_col is None:
        raise ValueError(
            f"No timestamp column found in {tel_path}. Columns: {list(tel.columns)[:20]}"
        )
    ts = pd.to_datetime(tel[ts_col], utc=True, errors="coerce")
    if ts.isna().any():
        print(f"  WARN: {int(ts.isna().sum())} unparsable timestamps — dropping those rows")
        tel = tel[ts.notna()].reset_index(drop=True)
        ts = ts[ts.notna()].reset_index(drop=True)
    timestamps_ms = (ts.astype("int64") // 1_000_000).to_numpy()

    # Every other numeric column is a telemetry channel
    channel_cols = [c for c in tel.columns if c != ts_col and pd.api.types.is_numeric_dtype(tel[c])]
    print(f"  channels: {len(channel_cols)}")
    if not channel_cols:
        raise ValueError(f"No numeric channel columns found in {tel_path}")
    telemetry_arr = tel[channel_cols].to_numpy(dtype=np.float32)

    # Anomaly file
    try:
        anom_path = _find_file(raw, _ANOMALY_FILENAMES)
        print(f"  anomalies: {anom_path}")
        anom = pd.read_csv(anom_path)
        start_col = next((c for c in _ANOMALY_START_COLUMNS if c in anom.columns), None)
        end_col = next((c for c in _ANOMALY_END_COLUMNS if c in anom.columns), None)
        if start_col is None or end_col is None:
            raise ValueError(
                f"No start/end columns in {anom_path}. Columns: {list(anom.columns)}"
            )
        starts = pd.to_datetime(anom[start_col], utc=True).astype("int64") // 1_000_000
        ends = pd.to_datetime(anom[end_col], utc=True).astype("int64") // 1_000_000
        anomaly_ranges = list(zip(starts.tolist(), ends.tolist(), strict=False))
        print(f"  anomaly intervals: {len(anomaly_ranges)}")
    except FileNotFoundError:
        print("  WARN: no anomaly file found — dataset will have empty labels")
        anomaly_ranges = []

    return timestamps_ms, telemetry_arr, channel_cols, anomaly_ranges


def _labels_from_ranges(timestamps_ms: np.ndarray, ranges: list) -> np.ndarray:
    labels = np.zeros(timestamps_ms.shape[0], dtype=np.bool_)
    for start_ms, end_ms in ranges:
        mask = (timestamps_ms >= start_ms) & (timestamps_ms < end_ms)
        labels[mask] = True
    return labels


def _sample_rate_hz(timestamps_ms: np.ndarray) -> float:
    if timestamps_ms.size < 2:
        return 1.0
    dt_ms = float(np.median(np.diff(timestamps_ms)))
    return 1000.0 / dt_ms if dt_ms > 0 else 1.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True, help="Directory holding raw OPS-SAT files")
    p.add_argument("--output", type=Path, default=Path("data/ops_sat"),
                   help="Output parquet directory (default: data/ops_sat)")
    p.add_argument("--downsample", type=int, default=1,
                   help="Take every N-th sample (default 1 = no downsample)")
    p.add_argument("--channels", default=None,
                   help="Comma-separated subset of channel names to keep (default: all)")
    p.add_argument("--name", default="ops_sat", help="Dataset name recorded in manifest.yaml")
    args = p.parse_args()

    args.raw = args.raw.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw OPS-SAT from: {args.raw}")
    timestamps_ms, telemetry_arr, channel_names, anomaly_ranges = _read_ops_sat_raw(args.raw)

    if args.downsample > 1:
        print(f"Downsampling by factor {args.downsample}")
        timestamps_ms = timestamps_ms[::args.downsample]
        telemetry_arr = telemetry_arr[::args.downsample]

    if args.channels:
        wanted = set(args.channels.split(","))
        keep = [i for i, name in enumerate(channel_names) if name in wanted]
        if not keep:
            raise SystemExit(f"None of the requested channels {wanted} found in dataset")
        channel_names = [channel_names[i] for i in keep]
        telemetry_arr = telemetry_arr[:, keep]
        print(f"Kept {len(keep)} channels: {channel_names[:10]}{'...' if len(keep) > 10 else ''}")

    labels = _labels_from_ranges(timestamps_ms, anomaly_ranges)
    sample_rate_hz = _sample_rate_hz(timestamps_ms)

    # ── data.parquet ──────────────────────────────────────────────────
    data_cols: dict = {"timestamp_ms": pa.array(timestamps_ms, type=pa.int64())}
    for i, name in enumerate(channel_names):
        data_cols[name] = pa.array(telemetry_arr[:, i], type=pa.float32())
    pq.write_table(pa.table(data_cols), args.output / "data.parquet")

    # ── labels.parquet ────────────────────────────────────────────────
    pq.write_table(
        pa.table({
            "timestamp_ms": pa.array(timestamps_ms, type=pa.int64()),
            "is_anomaly": pa.array(labels, type=pa.bool_()),
        }),
        args.output / "labels.parquet",
    )

    # ── channels.yaml ─────────────────────────────────────────────────
    (args.output / "channels.yaml").write_text(
        yaml.safe_dump({
            "channels": [
                {
                    "name": name,
                    "units": "",       # OPS-SAT rarely publishes units per channel
                    "subsystem": "unknown",
                    "sample_rate_hz": float(sample_rate_hz),
                    "description": f"OPS-SAT channel {name}",
                }
                for name in channel_names
            ]
        }, sort_keys=False),
    )

    # ── fault_log.json ────────────────────────────────────────────────
    (args.output / "fault_log.json").write_text(
        json.dumps(
            [{"type": "ops_sat_anomaly", "start_ms": int(s), "end_ms": int(e)}
             for s, e in anomaly_ranges],
            indent=2,
        ),
    )

    # ── manifest.yaml ─────────────────────────────────────────────────
    (args.output / "manifest.yaml").write_text(yaml.safe_dump({
        "name": args.name,
        "subsystem": "unknown",
        "sample_rate_hz": float(sample_rate_hz),
        "n_samples": int(telemetry_arr.shape[0]),
        "n_channels": int(telemetry_arr.shape[1]),
        "n_anomaly_intervals": len(anomaly_ranges),
        "n_anomaly_steps": int(labels.sum()),
        "anomaly_rate": float(labels.mean()),
        "source": "ESA OPS-SAT (converted via scripts/convert_ops_sat_to_parquet.py)",
        "raw_input": str(args.raw),
        "downsample_factor": args.downsample,
    }, sort_keys=False))

    print("\n-- Conversion complete --")
    print(f"output_dir:        {args.output}")
    print(f"n_samples:         {telemetry_arr.shape[0]}")
    print(f"n_channels:        {telemetry_arr.shape[1]}")
    print(f"sample_rate_hz:    {sample_rate_hz:.4f}")
    print(f"anomaly_intervals: {len(anomaly_ranges)}")
    print(f"anomaly_rate:      {float(labels.mean()):.4%}")
    print("\nUse it via: dataset=cached/ops_sat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
