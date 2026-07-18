"""ParquetTelemetryCorpus — union of many parquet-materialized datasets.

Reads N `ParquetTelemetryDataset` instances that share a channel schema
and presents them as a single virtual TelemetryDataset. This is the
"pretraining corpus" primitive: give Sat-TSFM one dataset argument that
contains multiple missions worth of data.

Design assumptions
------------------
* All member datasets MUST expose the same channel names, units, and
  subsystems, in the same order. The channel schema is asserted at
  construction time — mismatch raises ValueError immediately (fail loud,
  fail early — silent broadcast/reorder would corrupt training).
* Windows are yielded member-by-member. There is no cross-member window
  (i.e. no synthetic edge from the end of member A into the start of
  member B), which preserves the temporal integrity of each mission.
* Splits are computed per-member and then concatenated: a `train` split
  is every member's train segment, a `val` split is every member's val
  segment, etc. This keeps class balance similar across splits even if
  members have wildly different lengths.
* `stats()` aggregates over the concatenated full data (all splits, all
  members), giving normalization constants that reflect the corpus as a
  whole rather than any one mission.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from dgx_ts_core.data import (
    Channel,
    DatasetStats,
    SplitScheme,
    Subsystem,
    TelemetryWindow,
)
from dgx_ts_core.registry import DATASET_REGISTRY

from .parquet_telemetry import ParquetTelemetryDataset


def _channels_match(a: tuple[Channel, ...], b: tuple[Channel, ...]) -> bool:
    """True iff two channel tuples are the same schema (name+units+subsystem+rate)."""
    if len(a) != len(b):
        return False
    for ca, cb in zip(a, b, strict=True):
        if (
            ca.name != cb.name
            or ca.units != cb.units
            or ca.subsystem != cb.subsystem
            or float(ca.sample_rate_hz) != float(cb.sample_rate_hz)
        ):
            return False
    return True


class ParquetTelemetryCorpus:
    """Virtual TelemetryDataset backed by many ParquetTelemetryDataset members.

    Parameters
    ----------
    data_paths
        List of directories, each written by `dgx-ts synth`. Order is
        preserved in window iteration.
    name
        Human-readable identifier for logging / MLflow. Defaults to
        "corpus_of_<N>".
    subsystem
        Subsystem the corpus targets. Defaults to the first member's
        subsystem.
    """

    def __init__(
        self,
        data_paths: list[str] | list[Path],
        name: str | None = None,
        subsystem: str | Subsystem | None = None,
    ) -> None:
        if not data_paths:
            raise ValueError("ParquetTelemetryCorpus requires at least one data_path")

        # Load every member — a missing path raises FileNotFoundError with
        # the parquet loader's usual explanatory message.
        members: list[ParquetTelemetryDataset] = [
            ParquetTelemetryDataset(data_path=str(p)) for p in data_paths
        ]

        # Homogeneous channel schema — the pretrainer expects one shape.
        ref_channels = members[0].channels
        for i, m in enumerate(members[1:], start=1):
            if not _channels_match(ref_channels, m.channels):
                raise ValueError(
                    f"corpus member {i} ({m.name}) has channel schema that "
                    f"differs from member 0 ({members[0].name}); "
                    f"corpus training requires identical channels across members"
                )

        self._members = members
        self._channels = ref_channels
        self._sample_rate_hz = members[0].sample_rate_hz
        self._name = name or f"corpus_of_{len(members)}"

        if subsystem is None:
            self._subsystem = members[0].subsystem
        elif isinstance(subsystem, str):
            self._subsystem = Subsystem(subsystem)
        else:
            self._subsystem = subsystem

    # ── TelemetryDataset Protocol ────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def subsystem(self) -> Subsystem:
        return self._subsystem

    @property
    def channels(self) -> tuple[Channel, ...]:
        return self._channels

    @property
    def sample_rate_hz(self) -> float:
        return self._sample_rate_hz

    @property
    def has_labels(self) -> bool:
        # True as long as at least one member has labels — parquet loader
        # gives a zeros vector when labels.parquet is absent, so this is
        # always True in practice for us.
        return True

    def windows(self, length: int, stride: int) -> Iterator[TelemetryWindow]:
        """Iterate windows from every member sequentially.

        Provenance carries the corpus name and the member index so
        downstream consumers can attribute a window back to its origin.
        """
        for idx, member in enumerate(self._members):
            for w in member.windows(length, stride):
                # Preserve original provenance, then decorate with corpus context.
                merged = dict(w.provenance)
                merged.setdefault("member_index", idx)
                merged.setdefault("member_name", member.name)
                merged["corpus"] = self._name
                yield TelemetryWindow(
                    tensor=w.tensor,
                    timestamps=w.timestamps,
                    channels=w.channels,
                    labels=w.labels,
                    provenance=merged,
                )

    def split(self, scheme: SplitScheme) -> Mapping[str, ParquetTelemetryCorpus]:
        """Per-member split, then re-wrap each partition as a smaller corpus.

        The returned "train"/"val"/"test" corpora contain the corresponding
        partition of every original member — keeps mission diversity in every
        split rather than dumping whole missions into a single one.
        """
        per_member_splits = [m.split(scheme) for m in self._members]
        result: dict[str, ParquetTelemetryCorpus] = {}
        for split_name in ("train", "val", "test"):
            # Reuse the split ParquetTelemetryDataset objects directly by
            # wrapping them into a new corpus — the ctor above only requires
            # data_paths, so we need a small alternate ctor path. Use the
            # `_from_members` classmethod below to avoid re-reading parquet.
            partition_members = [s[split_name] for s in per_member_splits]
            result[split_name] = ParquetTelemetryCorpus._from_members(
                members=partition_members,
                name=f"{self._name}_{split_name}",
                subsystem=self._subsystem,
            )
        return result

    @classmethod
    def _from_members(
        cls,
        members: list[ParquetTelemetryDataset],
        name: str,
        subsystem: Subsystem,
    ) -> ParquetTelemetryCorpus:
        """Internal ctor that skips path-loading — used by split()."""
        obj = cls.__new__(cls)
        obj._members = members
        obj._channels = members[0].channels
        obj._sample_rate_hz = members[0].sample_rate_hz
        obj._name = name
        obj._subsystem = subsystem
        return obj

    def stats(self) -> DatasetStats:
        """Aggregate stats across every member's full data.

        Means/stds/p99 are computed over the concatenated raw data so
        normalization is corpus-wide, not per-mission.
        """
        # Concatenate the (n_samples, n_channels) matrices from every member.
        # We access the internal _data attr directly to avoid re-materializing
        # windows just for stats.
        all_data = np.concatenate([m._data for m in self._members], axis=0)  # noqa: SLF001
        return DatasetStats(
            means=all_data.mean(axis=0).astype(np.float32),
            stds=(all_data.std(axis=0) + 1e-8).astype(np.float32),
            p99=np.quantile(np.abs(all_data), 0.99, axis=0).astype(np.float32),
            n_samples=int(all_data.shape[0]),
            n_channels=int(all_data.shape[1]),
        )


@DATASET_REGISTRY.register("parquet_telemetry_corpus")
def _create(**kwargs: Any) -> ParquetTelemetryCorpus:
    return ParquetTelemetryCorpus(**kwargs)
