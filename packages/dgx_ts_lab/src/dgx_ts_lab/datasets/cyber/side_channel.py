"""SideChannelDataset — generic behavior-fingerprint adapter.

Wraps ANY existing `TelemetryDataset` and derives a new dataset where each
sample is a vector of rolling-window statistics over the source telemetry.
Lets you apply behavior-style AD to EPS, ADCS, or any subsystem without
writing a new dataset class per subsystem.

Default summary function computes per-channel:
    mean, std, max-abs, energy (sum of squares)

Override via the ``summary_fn`` kwarg for custom behavior fingerprints.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
from dgx_ts_core.data import (
    Channel,
    DatasetStats,
    SplitScheme,
    Subsystem,
    TelemetryDataset,
    TelemetryWindow,
    Units,
)
from dgx_ts_core.registry import DATASET_REGISTRY

DEFAULT_FEATURE_FUNCS = ("mean", "std", "max_abs", "energy")


def _default_summary(
    arr: np.ndarray, feature_funcs: tuple[str, ...] = DEFAULT_FEATURE_FUNCS
) -> np.ndarray:
    """Per-channel stats over a window of shape (T, C). Returns (C*len(funcs),)."""
    out = []
    for name in feature_funcs:
        if name == "mean":
            out.append(arr.mean(axis=0))
        elif name == "std":
            out.append(arr.std(axis=0))
        elif name == "max_abs":
            out.append(np.abs(arr).max(axis=0))
        elif name == "energy":
            out.append((arr ** 2).sum(axis=0))
        else:
            raise ValueError(f"unknown feature func: {name}")
    return np.concatenate(out).astype(np.float32)


class SideChannelDataset:
    """Behavior-fingerprint view of any TelemetryDataset."""

    def __init__(
        self,
        source: TelemetryDataset | None = None,
        summary_window_length: int = 60,
        summary_stride: int = 30,
        feature_funcs: tuple[str, ...] = DEFAULT_FEATURE_FUNCS,
        summary_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        name: str | None = None,
        _materialized: dict | None = None,
    ) -> None:
        if _materialized is not None:
            self._data = _materialized["data"]
            self._labels = _materialized["labels"]
            self._channels = _materialized["channels"]
            self._sample_rate_hz = _materialized["sample_rate_hz"]
            self._subsystem = _materialized["subsystem"]
            self._name = _materialized["name"]
            self._timestamps = _materialized["timestamps"]
            return

        if source is None:
            raise ValueError("source TelemetryDataset is required (or pass _materialized)")
        self._src_name = source.name
        self._subsystem = source.subsystem
        self._sample_rate_hz = source.sample_rate_hz / max(summary_stride, 1)
        self._name = name or f"sidechannel[{source.name}]"
        self._summary_fn = summary_fn or (
            lambda arr: _default_summary(arr, feature_funcs)
        )

        # Walk the source's windows to build behavior-feature vectors.
        feature_rows: list[np.ndarray] = []
        any_label_rows: list[bool] = []
        for w in source.windows(length=summary_window_length, stride=summary_stride):
            feature_rows.append(self._summary_fn(w.tensor))
            any_label_rows.append(
                bool(w.labels.any()) if w.labels is not None else False
            )
        if not feature_rows:
            raise ValueError(
                f"source {source.name} produced no windows at length={summary_window_length}, "
                f"stride={summary_stride}"
            )
        self._data = np.stack(feature_rows, axis=0).astype(np.float32)
        self._labels = np.asarray(any_label_rows, dtype=np.bool_)
        n_features = self._data.shape[1]
        self._channels = tuple(
            Channel(
                name=f"sc_{i}",
                units=Units.DIMENSIONLESS,
                subsystem=self._subsystem,
                sample_rate_hz=self._sample_rate_hz,
                description=f"Side-channel summary feature {i}",
            )
            for i in range(n_features)
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

    def split(self, scheme: SplitScheme) -> Mapping[str, SideChannelDataset]:
        n = len(self._data)
        n_train = int(scheme.train_frac * n)
        n_val = int(scheme.val_frac * n)
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n),
        }
        return {
            split_name: SideChannelDataset(
                _materialized={
                    "data": self._data[s:e].copy(),
                    "labels": self._labels[s:e].copy(),
                    "channels": self._channels,
                    "sample_rate_hz": self._sample_rate_hz,
                    "subsystem": self._subsystem,
                    "name": f"{self._name}_{split_name}",
                    "timestamps": self._timestamps[s:e].copy(),
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


@DATASET_REGISTRY.register("side_channel")
def _create(**kwargs: Any) -> SideChannelDataset:
    return SideChannelDataset(**kwargs)
