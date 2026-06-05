"""ActivityWindowDataset — rolling statistical features over operator activity.

Each "window" sample is a vector of behavior statistics computed over a
rolling window of recent operator actions:

    login_freq        logins per hour
    command_rate      commands per minute
    command_diversity Shannon entropy over command types
    hour_of_day_sin   cyclic encoding of time-of-day
    hour_of_day_cos   "
    session_length    current session duration (minutes)

This is exactly what the OperatorFingerprintModel consumes. Aux labels
``{"operator_id": int}`` per timestep let the model know which operator's
distribution to compare against.
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


ACTIVITY_FEATURE_NAMES = (
    "login_freq",
    "command_rate",
    "command_diversity",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "session_length",
)


class ActivityWindowDataset:
    """A TelemetryDataset over per-timestep operator activity statistics."""

    def __init__(
        self,
        features: np.ndarray,                # (N, F) float32 — F == len(ACTIVITY_FEATURE_NAMES)
        operator_ids: np.ndarray,            # (N,) int64
        labels: np.ndarray | None = None,    # (N,) bool — True = impersonation injection
        sample_rate_hz: float = 1.0 / 60.0,  # default: 1 sample per minute
        name: str = "activity_window",
        subsystem: Subsystem = Subsystem.OBDH,
        _materialized: dict | None = None,
    ) -> None:
        if _materialized is not None:
            self._data = _materialized["data"]
            self._operator_ids = _materialized["operator_ids"]
            self._labels = _materialized["labels"]
            self._sample_rate_hz = float(_materialized["sample_rate_hz"])
            self._name = _materialized["name"]
            self._subsystem = _materialized["subsystem"]
        else:
            if features.shape[1] != len(ACTIVITY_FEATURE_NAMES):
                raise ValueError(
                    f"features must have {len(ACTIVITY_FEATURE_NAMES)} columns "
                    f"(got {features.shape[1]}). Expected: {ACTIVITY_FEATURE_NAMES}"
                )
            self._data = features.astype(np.float32)
            self._operator_ids = operator_ids.astype(np.int64)
            self._labels = (
                labels.astype(np.bool_)
                if labels is not None
                else np.zeros(len(features), dtype=np.bool_)
            )
            self._sample_rate_hz = float(sample_rate_hz)
            self._name = name
            self._subsystem = subsystem

        self._channels = tuple(
            Channel(
                name=fname,
                units=Units.DIMENSIONLESS,
                subsystem=self._subsystem,
                sample_rate_hz=self._sample_rate_hz,
                description=f"Activity-window feature: {fname}",
            )
            for fname in ACTIVITY_FEATURE_NAMES
        )
        self._timestamps = np.arange(len(self._data), dtype=np.int64) * int(
            1000 / max(self._sample_rate_hz, 1e-6)
        )

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

    @property
    def n_operators(self) -> int:
        return int(self._operator_ids.max()) + 1

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
                aux_labels={"operator_id": self._operator_ids[start:end].copy()},
                provenance={"source": self._name, "start": start, "end": end},
            )

    def split(self, scheme: SplitScheme) -> Mapping[str, "ActivityWindowDataset"]:
        n = len(self._data)
        n_train = int(scheme.train_frac * n)
        n_val = int(scheme.val_frac * n)
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n),
        }
        return {
            split_name: ActivityWindowDataset(
                features=self._data[s:e].copy(),
                operator_ids=self._operator_ids[s:e].copy(),
                labels=self._labels[s:e].copy(),
                sample_rate_hz=self._sample_rate_hz,
                name=f"{self._name}_{split_name}",
                subsystem=self._subsystem,
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


@DATASET_REGISTRY.register("activity_window")
def _create(**kwargs: Any) -> ActivityWindowDataset:
    return ActivityWindowDataset(**kwargs)
