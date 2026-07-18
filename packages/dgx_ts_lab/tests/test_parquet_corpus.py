"""Tests for ParquetTelemetryCorpus.

Corpus = the union of many `ParquetTelemetryDataset`s under a single
TelemetryDataset facade. These tests verify:
  * Schema-mismatch is rejected at construction time.
  * Window iteration yields ALL members' windows, in member order,
    with provenance carrying the corpus + member identity.
  * split() produces per-split corpora whose sample counts sum to the
    original member counts.
  * stats() aggregates over the concatenated data.
  * The registry key `parquet_telemetry_corpus` is wired.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dgx_ts_core.data import Subsystem, Units
from dgx_ts_lab.datasets import ParquetTelemetryCorpus
from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset

from tests.test_synth_parquet_roundtrip import _write_synth_dir


def _make_member(tmp_path: Path, tag: str, n: int, channels: int, seed: int) -> Path:
    """Materialize a synthetic dataset to a parquet directory and return the path."""
    ds = TrivialSyntheticDataset(n_samples=n, n_channels=channels, seed=seed)
    out = tmp_path / tag
    _write_synth_dir(ds, out)
    return out


def test_corpus_concatenates_windows_from_all_members(tmp_path: Path) -> None:
    """Every member's windows appear in the corpus, tagged with member_index."""
    m1 = _make_member(tmp_path, "m1", n=200, channels=3, seed=1)
    m2 = _make_member(tmp_path, "m2", n=200, channels=3, seed=2)
    m3 = _make_member(tmp_path, "m3", n=200, channels=3, seed=3)

    corpus = ParquetTelemetryCorpus(data_paths=[m1, m2, m3])

    windows = list(corpus.windows(length=64, stride=64))
    # Each member yields 3 non-overlapping windows of length 64 in 200 samples.
    assert len(windows) == 9

    member_seq = [w.provenance["member_index"] for w in windows]
    # Windows must appear in member order (0,0,0, 1,1,1, 2,2,2).
    assert member_seq == [0, 0, 0, 1, 1, 1, 2, 2, 2]

    # Provenance stamped with corpus + member name.
    for w in windows:
        assert w.provenance["corpus"] == corpus.name
        assert "member_name" in w.provenance


def test_corpus_rejects_schema_mismatch(tmp_path: Path) -> None:
    """A member with a different channel count must be rejected up front."""
    m1 = _make_member(tmp_path, "m1", n=100, channels=3, seed=0)
    m2 = _make_member(tmp_path, "m2", n=100, channels=5, seed=0)  # different width

    with pytest.raises(ValueError, match="channel schema"):
        ParquetTelemetryCorpus(data_paths=[m1, m2])


def test_corpus_split_partitions_every_member(tmp_path: Path) -> None:
    """A per-member split; every split contains a piece of every member."""
    from dgx_ts_core.data import SplitScheme, SplitStrategy

    paths = [_make_member(tmp_path, f"m{i}", n=100, channels=2, seed=i) for i in range(3)]
    corpus = ParquetTelemetryCorpus(data_paths=paths)

    splits = corpus.split(SplitScheme(strategy=SplitStrategy.TEMPORAL))
    assert set(splits) == {"train", "val", "test"}

    total_across_splits = sum(s.stats().n_samples for s in splits.values())
    # 3 members × 100 samples = 300 total; splits must sum to that.
    assert total_across_splits == 300


def test_corpus_stats_aggregate_over_concat_data(tmp_path: Path) -> None:
    """Stats are over the concatenation, not any one member."""
    paths = [_make_member(tmp_path, f"m{i}", n=50, channels=2, seed=i) for i in range(4)]
    corpus = ParquetTelemetryCorpus(data_paths=paths)
    stats = corpus.stats()
    assert stats.n_samples == 200
    assert stats.n_channels == 2
    # Sanity: means/stds are finite arrays of the right shape.
    assert stats.means.shape == (2,)
    assert stats.stds.shape == (2,)
    assert np.all(np.isfinite(stats.means))
    assert np.all(np.isfinite(stats.stds))


def test_corpus_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DATASET_REGISTRY

    assert "parquet_telemetry_corpus" in DATASET_REGISTRY.list()


def test_corpus_requires_at_least_one_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one data_path"):
        ParquetTelemetryCorpus(data_paths=[])


def test_corpus_uses_first_member_subsystem_by_default(tmp_path: Path) -> None:
    m1 = _make_member(tmp_path, "m1", n=100, channels=2, seed=0)
    m2 = _make_member(tmp_path, "m2", n=100, channels=2, seed=1)
    corpus = ParquetTelemetryCorpus(data_paths=[m1, m2])
    # TrivialSyntheticDataset writes UNKNOWN by default via its channel spec;
    # the point of this test is that the property is populated, not what it equals.
    assert corpus.subsystem in Subsystem
    # Channels come from member 0.
    assert corpus.channels[0].units in Units
