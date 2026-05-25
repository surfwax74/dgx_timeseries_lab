"""Tests for the NASA Telemanom loader.

Uses a synthetic fixture so the test runs anywhere, including the
air-gapped DGX where the real NASA files may or may not be present.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from dgx_ts_core.data import SplitScheme, SplitStrategy, TelemetryDataset
from dgx_ts_lab.datasets import NasaTelemanomChannel


@pytest.fixture
def fake_telemanom_root(tmp_path: Path) -> Path:
    """Build a tiny Telemanom-formatted directory in tmp_path."""
    root = tmp_path / "fake_telemanom"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir(parents=True)

    rng = np.random.default_rng(0)
    # T-1: 800 train values, 600 test values, anomaly at [100, 150]
    train = rng.normal(0, 1, size=(800, 1)).astype(np.float32)
    test = rng.normal(0, 1, size=(600, 1)).astype(np.float32)
    test[100:151, 0] += 10.0  # injected anomaly
    np.save(root / "train" / "T-1.npy", train)
    np.save(root / "test" / "T-1.npy", test)

    # labeled_anomalies.csv with one row
    with open(root / "labeled_anomalies.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["chan_id", "spacecraft", "anomaly_sequences", "class", "num_values"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "chan_id": "T-1",
                "spacecraft": "SMAP",
                "anomaly_sequences": json.dumps([[100, 150]]),
                "class": "[point]",
                "num_values": 600,
            }
        )
    return root


def test_loader_implements_protocol(fake_telemanom_root: Path) -> None:
    ds = NasaTelemanomChannel(
        data_root=fake_telemanom_root, channel_id="T-1", spacecraft="SMAP"
    )
    assert isinstance(ds, TelemetryDataset)
    assert ds.name == "nasa_smap_T-1"
    assert ds.has_labels
    assert len(ds.channels) == 1


def test_loader_marks_anomalies_correctly(fake_telemanom_root: Path) -> None:
    ds = NasaTelemanomChannel(
        data_root=fake_telemanom_root, channel_id="T-1", spacecraft="SMAP"
    )
    # Train labels should be all False
    assert not ds._train_labels.any()
    # Test labels should mark [100, 151) as anomaly
    assert ds._test_labels[100:151].all()
    assert not ds._test_labels[:100].any()
    assert not ds._test_labels[151:].any()


def test_loader_split_returns_canonical_partition(fake_telemanom_root: Path) -> None:
    ds = NasaTelemanomChannel(
        data_root=fake_telemanom_root, channel_id="T-1", spacecraft="SMAP"
    )
    splits = ds.split(SplitScheme(strategy=SplitStrategy.TEMPORAL))
    train, val, test = splits["train"], splits["val"], splits["test"]
    # Train uses train.npy
    assert train._train_data.shape[0] == 800
    # Val + test together reconstruct test.npy (600 samples)
    assert val._train_data.shape[0] + test._train_data.shape[0] == 600


def test_loader_missing_files_raises_with_pointer(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="github.com/khundman/telemanom"):
        NasaTelemanomChannel(data_root=empty, channel_id="X-1", spacecraft="SMAP")


def test_loader_unknown_channel_raises(fake_telemanom_root: Path) -> None:
    # Provide a npy for an extra channel so we get past the file check,
    # then fail at CSV lookup.
    np.save(fake_telemanom_root / "train" / "Z-99.npy", np.zeros((10, 1), dtype=np.float32))
    np.save(fake_telemanom_root / "test" / "Z-99.npy", np.zeros((10, 1), dtype=np.float32))
    with pytest.raises(KeyError, match="Z-99"):
        NasaTelemanomChannel(
            data_root=fake_telemanom_root, channel_id="Z-99", spacecraft="SMAP"
        )


def test_registry_keys_present() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DATASET_REGISTRY

    assert "nasa_smap_channel" in DATASET_REGISTRY.list()
    assert "nasa_msl_channel" in DATASET_REGISTRY.list()
