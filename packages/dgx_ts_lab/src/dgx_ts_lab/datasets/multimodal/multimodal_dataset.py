"""MultiModalDataset — TelemetryDataset over (telemetry, commands, logs) jointly.

Tensor layout: concatenated along the channel dim, in fixed order:

    [telemetry channels (C_tel) | command features (3) | log features (3)]

``channel_modalities`` is a parallel tuple telling the model which group
each channel belongs to. Reused by SatMultiModalModule's sub-encoder split.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from dgx_ts_core.data import (
    Channel,
    DatasetStats,
    SplitScheme,
    Subsystem,
    TelemetryWindow,
    Units,
)
from dgx_ts_core.registry import DATASET_REGISTRY


COMMAND_FEATURE_NAMES = ("cmd_count", "cmd_last_opcode_id", "cmd_last_param_id")
LOG_FEATURE_NAMES = ("log_count", "log_max_severity", "log_last_source_id")


class MultiModalDataset:
    """A TelemetryDataset that bundles three aligned 1-Hz modalities."""

    def __init__(
        self,
        telemetry: np.ndarray,            # (T, C_tel) float32
        commands: np.ndarray,             # (T, 3) float32 — from CommandEventBucketer
        logs: np.ndarray,                 # (T, 3) float32 — from LogEventBucketer
        telemetry_channels: tuple[Channel, ...],
        sample_rate_hz: float = 1.0,
        labels: np.ndarray | None = None,
        name: str = "multimodal",
        subsystem: Subsystem = Subsystem.UNKNOWN,
        _materialized: dict | None = None,
    ) -> None:
        if _materialized is not None:
            self._data = _materialized["data"]
            self._labels = _materialized["labels"]
            self._timestamps = _materialized["timestamps"]
            self._channels = _materialized["channels"]
            self._channel_modalities = _materialized["channel_modalities"]
            self._sample_rate_hz = _materialized["sample_rate_hz"]
            self._name = _materialized["name"]
            self._subsystem = _materialized["subsystem"]
            self._c_tel = _materialized["c_tel"]
            return

        if telemetry.shape[0] != commands.shape[0] or telemetry.shape[0] != logs.shape[0]:
            raise ValueError(
                f"all modalities must align in time: telemetry={telemetry.shape}, "
                f"commands={commands.shape}, logs={logs.shape}"
            )
        if commands.shape[1] != 3 or logs.shape[1] != 3:
            raise ValueError("commands and logs must have exactly 3 features each")
        if len(telemetry_channels) != telemetry.shape[1]:
            raise ValueError(
                f"telemetry_channels length {len(telemetry_channels)} != "
                f"telemetry.shape[1] {telemetry.shape[1]}"
            )

        self._sample_rate_hz = float(sample_rate_hz)
        self._name = name
        self._subsystem = subsystem
        self._c_tel = telemetry.shape[1]

        # Concatenate modalities along channel dim
        self._data = np.concatenate([telemetry, commands, logs], axis=1).astype(np.float32)
        self._labels = (
            labels.astype(np.bool_) if labels is not None
            else np.zeros(telemetry.shape[0], dtype=np.bool_)
        )
        self._timestamps = (
            np.arange(telemetry.shape[0], dtype=np.int64)
            * int(1000 / max(self._sample_rate_hz, 1e-6))
        )

        # Build channel list + modality mapping
        cmd_channels = tuple(
            Channel(
                name=fname,
                units=Units.DIMENSIONLESS,
                subsystem=Subsystem.OBDH,
                sample_rate_hz=self._sample_rate_hz,
                description=f"Command event feature: {fname}",
            )
            for fname in COMMAND_FEATURE_NAMES
        )
        log_channels = tuple(
            Channel(
                name=fname,
                units=Units.DIMENSIONLESS,
                subsystem=Subsystem.OBDH,
                sample_rate_hz=self._sample_rate_hz,
                description=f"Log event feature: {fname}",
            )
            for fname in LOG_FEATURE_NAMES
        )
        self._channels = tuple(telemetry_channels) + cmd_channels + log_channels
        self._channel_modalities = (
            ("telemetry",) * self._c_tel + ("command",) * 3 + ("log",) * 3
        )

    # ── Modality accessors ──────────────────────────────────────────────

    @property
    def n_telemetry_channels(self) -> int:
        return self._c_tel

    @property
    def channel_modalities(self) -> tuple[str, ...]:
        return self._channel_modalities

    def split_window_by_modality(self, window: TelemetryWindow) -> dict[str, np.ndarray]:
        """Slice a window's tensor into the three modality views."""
        T = window.tensor.shape[0]
        return {
            "telemetry": window.tensor[:, : self._c_tel],                         # (T, C_tel)
            "commands": window.tensor[:, self._c_tel : self._c_tel + 3],          # (T, 3)
            "logs": window.tensor[:, self._c_tel + 3 : self._c_tel + 6],          # (T, 3)
        }

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
                provenance={
                    "source": self._name,
                    "start": start,
                    "end": end,
                    "channel_modalities": self._channel_modalities,
                    "n_telemetry_channels": self._c_tel,
                },
            )

    def split(self, scheme: SplitScheme) -> Mapping[str, "MultiModalDataset"]:
        n = len(self._data)
        n_train = int(scheme.train_frac * n)
        n_val = int(scheme.val_frac * n)
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n),
        }
        return {
            split_name: MultiModalDataset(
                telemetry=np.zeros((1, 1), dtype=np.float32),  # unused; _materialized takes over
                commands=np.zeros((1, 3), dtype=np.float32),
                logs=np.zeros((1, 3), dtype=np.float32),
                telemetry_channels=self._channels[: self._c_tel],
                _materialized={
                    "data": self._data[s:e].copy(),
                    "labels": self._labels[s:e].copy(),
                    "timestamps": self._timestamps[s:e].copy(),
                    "channels": self._channels,
                    "channel_modalities": self._channel_modalities,
                    "sample_rate_hz": self._sample_rate_hz,
                    "name": f"{self._name}_{split_name}",
                    "subsystem": self._subsystem,
                    "c_tel": self._c_tel,
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


@DATASET_REGISTRY.register("multimodal")
def _create(**kwargs: Any) -> MultiModalDataset:
    return MultiModalDataset(**kwargs)
