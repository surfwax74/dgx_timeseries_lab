"""ParquetTelemetryDataset — load datasets written by ``dgx-ts synth``.

The canonical air-gapped distribution pattern: generate once with synth,
distribute the parquet directory, load anywhere with this loader. No
re-generation, identical data on every machine.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml
from dgx_ts_core.data import (
    Channel,
    DatasetStats,
    SplitScheme,
    Subsystem,
    TelemetryWindow,
    Units,
)
from dgx_ts_core.registry import DATASET_REGISTRY

_EXPECTED_LAYOUT = """\
Expected layout at data_path:
    data.parquet      timestamp_ms + one column per channel
    labels.parquet    timestamp_ms + is_anomaly
    channels.yaml     channel metadata
    manifest.yaml     dataset metadata
    fault_log.json    (optional) structured fault entries

Generate with: dgx-ts synth dataset=<config> output_dir=<dir>
"""


class ParquetTelemetryDataset:
    """A TelemetryDataset materialized from a parquet directory."""

    def __init__(
        self,
        data_path: str | Path,
        sample_rate_hz: float | None = None,
        name: str | None = None,
        _materialized: dict[str, Any] | None = None,
    ) -> None:
        if _materialized is not None:
            self._data = _materialized["data"]
            self._timestamps = _materialized["timestamps"]
            self._labels = _materialized["labels"]
            self._channels = _materialized["channels"]
            self._sample_rate_hz = _materialized["sample_rate_hz"]
            self._name = _materialized["name"]
            self._subsystem = _materialized.get("subsystem", Subsystem.UNKNOWN)
            return

        root = Path(data_path)
        if not root.exists():
            raise FileNotFoundError(f"parquet dataset not found: {root}\n\n{_EXPECTED_LAYOUT}")

        # Manifest
        manifest_path = root / "manifest.yaml"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f) or {}
        self._name = name or str(manifest.get("name", root.name))
        sub_val = manifest.get("subsystem", "unknown")
        self._subsystem = Subsystem(sub_val) if isinstance(sub_val, str) else sub_val
        self._sample_rate_hz = float(
            sample_rate_hz if sample_rate_hz is not None else manifest.get("sample_rate_hz", 1.0)
        )

        # Channels
        channels_path = root / "channels.yaml"
        if not channels_path.exists():
            raise FileNotFoundError(f"channels.yaml missing in {root}\n\n{_EXPECTED_LAYOUT}")
        with open(channels_path) as f:
            ch_doc = yaml.safe_load(f) or {}
        ch_objs: list[Channel] = []
        for c in ch_doc.get("channels", []):
            ch_objs.append(
                Channel(
                    name=c["name"],
                    units=Units(c["units"]) if isinstance(c["units"], str) else c["units"],
                    subsystem=Subsystem(c["subsystem"])
                    if isinstance(c["subsystem"], str)
                    else c["subsystem"],
                    sample_rate_hz=float(c["sample_rate_hz"]),
                    description=c.get("description", ""),
                )
            )
        self._channels = tuple(ch_objs)

        # Data
        data_file = root / "data.parquet"
        if not data_file.exists():
            raise FileNotFoundError(f"data.parquet missing in {root}\n\n{_EXPECTED_LAYOUT}")
        table = pq.read_table(data_file)
        self._timestamps = table.column("timestamp_ms").to_numpy().astype(np.int64)
        self._data = np.column_stack(
            [table.column(ch.name).to_numpy().astype(np.float32) for ch in ch_objs]
        )

        # Labels (optional)
        labels_file = root / "labels.parquet"
        if labels_file.exists():
            lbl_table = pq.read_table(labels_file)
            self._labels = lbl_table.column("is_anomaly").to_numpy().astype(np.bool_)
        else:
            self._labels = np.zeros(len(self._data), dtype=np.bool_)

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
        return True

    def windows(self, length: int, stride: int) -> Iterator[TelemetryWindow]:
        n = len(self._data)
        if length > n:
            return
        for start in range(0, n - length + 1, stride):
            end = start + length
            yield TelemetryWindow(
                tensor=self._data[start:end].copy(),
                timestamps=self._timestamps[start:end].copy(),
                channels=self._channels,
                labels=self._labels[start:end].copy(),
                provenance={"source": self._name, "start": start, "end": end},
            )

    def split(self, scheme: SplitScheme) -> Mapping[str, ParquetTelemetryDataset]:
        n = len(self._data)
        n_train = int(scheme.train_frac * n)
        n_val = int(scheme.val_frac * n)
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n),
        }
        return {
            split_name: ParquetTelemetryDataset(
                data_path="",  # unused on the materialized path
                _materialized={
                    "data": self._data[s:e].copy(),
                    "timestamps": self._timestamps[s:e].copy(),
                    "labels": self._labels[s:e].copy(),
                    "channels": self._channels,
                    "sample_rate_hz": self._sample_rate_hz,
                    "name": f"{self._name}_{split_name}",
                    "subsystem": self._subsystem,
                },
            )
            for split_name, (s, e) in bounds.items()
        }

    def stats(self) -> DatasetStats:
        return DatasetStats(
            means=self._data.mean(axis=0).astype(np.float32),
            stds=(self._data.std(axis=0) + 1e-8).astype(np.float32),
            p99=np.quantile(np.abs(self._data), 0.99, axis=0).astype(np.float32),
            n_samples=int(self._data.shape[0]),
            n_channels=int(self._data.shape[1]),
        )


@DATASET_REGISTRY.register("parquet_telemetry")
def _create(**kwargs: Any) -> ParquetTelemetryDataset:
    return ParquetTelemetryDataset(**kwargs)
