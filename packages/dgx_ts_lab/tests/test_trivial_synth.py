"""Unit tests for TrivialSyntheticDataset."""

from __future__ import annotations

import numpy as np

from dgx_ts_core.data import SplitScheme, SplitStrategy, TelemetryDataset
from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset


def test_dataset_implements_protocol() -> None:
    ds = TrivialSyntheticDataset(n_samples=500, n_channels=3, seed=0)
    assert isinstance(ds, TelemetryDataset)
    assert ds.has_labels
    assert len(ds.channels) == 3
    assert ds.sample_rate_hz == 1.0


def test_windows_have_correct_shape_and_labels() -> None:
    ds = TrivialSyntheticDataset(n_samples=200, n_channels=2, seed=1)
    windows = list(ds.windows(length=64, stride=32))
    assert len(windows) == (200 - 64) // 32 + 1
    for w in windows:
        assert w.tensor.shape == (64, 2)
        assert w.timestamps.shape == (64,)
        assert w.labels is not None and w.labels.shape == (64,)
        assert w.tensor.dtype == np.float32


def test_split_is_chronological_and_disjoint() -> None:
    ds = TrivialSyntheticDataset(n_samples=1000, n_channels=2, seed=2)
    splits = ds.split(
        SplitScheme(strategy=SplitStrategy.TEMPORAL, train_frac=0.7, val_frac=0.15, test_frac=0.15)
    )
    train, val, test = splits["train"], splits["val"], splits["test"]
    # Each is itself a TelemetryDataset.
    assert isinstance(train, TelemetryDataset)
    # Concatenated lengths reconstruct the original (within rounding).
    total = (
        train.stats().n_samples + val.stats().n_samples + test.stats().n_samples
    )
    assert total == 1000


def test_anomalies_are_present_and_labeled() -> None:
    ds = TrivialSyntheticDataset(
        n_samples=5000, n_channels=2, anomaly_rate=0.02, seed=3
    )
    # Pull all windows non-overlapping and count labels.
    labels = np.concatenate(
        [w.labels for w in ds.windows(length=100, stride=100) if w.labels is not None]
    )
    # Should see ~2% positives; allow a generous range for the rng.
    assert 0.5 < labels.mean() * 100 < 4.0


def test_stats_shape_matches_channels() -> None:
    ds = TrivialSyntheticDataset(n_samples=400, n_channels=4, seed=4)
    s = ds.stats()
    assert s.means.shape == (4,)
    assert s.stds.shape == (4,)
    assert s.p99.shape == (4,)
    assert s.n_channels == 4
    assert s.n_samples == 400


def test_determinism_via_seed() -> None:
    a = TrivialSyntheticDataset(n_samples=200, n_channels=2, seed=42)
    b = TrivialSyntheticDataset(n_samples=200, n_channels=2, seed=42)
    win_a = next(a.windows(length=50, stride=50))
    win_b = next(b.windows(length=50, stride=50))
    np.testing.assert_array_equal(win_a.tensor, win_b.tensor)
    np.testing.assert_array_equal(win_a.labels, win_b.labels)  # type: ignore[arg-type]
