"""End-to-end test: generate a dataset, write parquet, load back, verify identity.

This is the canonical air-gapped distribution test — it proves you can
generate-once-on-any-machine and load-anywhere with byte-identical results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from dgx_ts_lab.datasets import ParquetTelemetryDataset
from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset


def _write_synth_dir(ds, out_dir: Path) -> None:
    """Replicate what cli/synth.py does, minus the Hydra plumbing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = ds._data
    timestamps = ds._timestamps
    labels = ds._labels
    channels = ds.channels

    import pyarrow as pa

    pq.write_table(
        pa.table(
            {
                "timestamp_ms": pa.array(timestamps, type=pa.int64()),
                **{ch.name: pa.array(data[:, i], type=pa.float32()) for i, ch in enumerate(channels)},
            }
        ),
        out_dir / "data.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "timestamp_ms": pa.array(timestamps, type=pa.int64()),
                "is_anomaly": pa.array(labels, type=pa.bool_()),
            }
        ),
        out_dir / "labels.parquet",
    )
    (out_dir / "fault_log.json").write_text(
        json.dumps(list(getattr(ds, "fault_log", [])), indent=2, default=str)
    )
    (out_dir / "channels.yaml").write_text(
        yaml.safe_dump(
            {
                "channels": [
                    {
                        "name": ch.name,
                        "units": ch.units.value,
                        "subsystem": ch.subsystem.value,
                        "sample_rate_hz": float(ch.sample_rate_hz),
                        "description": ch.description,
                    }
                    for ch in channels
                ]
            },
            sort_keys=False,
        )
    )
    (out_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "name": ds.name,
                "subsystem": ds.subsystem.value,
                "sample_rate_hz": float(ds.sample_rate_hz),
                "n_samples": int(data.shape[0]),
                "n_channels": int(data.shape[1]),
            },
            sort_keys=False,
        )
    )


def test_synth_then_load_preserves_data(tmp_path: Path) -> None:
    src = TrivialSyntheticDataset(n_samples=500, n_channels=3, seed=42)
    out_dir = tmp_path / "synth_out"
    _write_synth_dir(src, out_dir)

    loaded = ParquetTelemetryDataset(data_path=out_dir)

    # Identical data.
    np.testing.assert_array_equal(loaded._data, src._data)
    np.testing.assert_array_equal(loaded._labels, src._labels)
    np.testing.assert_array_equal(loaded._timestamps, src._timestamps)
    assert len(loaded.channels) == len(src.channels)
    assert all(a.name == b.name for a, b in zip(loaded.channels, src.channels, strict=False))


def test_loaded_dataset_implements_protocol(tmp_path: Path) -> None:
    src = TrivialSyntheticDataset(n_samples=300, n_channels=2, seed=0)
    out_dir = tmp_path / "parquet"
    _write_synth_dir(src, out_dir)
    loaded = ParquetTelemetryDataset(data_path=out_dir)

    # Windows iterate.
    win = next(loaded.windows(length=64, stride=64))
    assert win.tensor.shape == (64, 2)
    assert win.labels is not None and win.labels.shape == (64,)

    # Split.
    from dgx_ts_core.data import SplitScheme, SplitStrategy

    splits = loaded.split(SplitScheme(strategy=SplitStrategy.TEMPORAL))
    assert set(splits) == {"train", "val", "test"}
    total = sum(s.stats().n_samples for s in splits.values())
    assert total == 300


def test_parquet_loader_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DATASET_REGISTRY

    assert "parquet_telemetry" in DATASET_REGISTRY.list()


def test_missing_parquet_dir_raises_with_pointer(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError, match="dgx-ts synth"):
        ParquetTelemetryDataset(data_path=tmp_path / "does_not_exist")
