"""Phase 0 trivial synthetic dataset: sine waves with injected point anomalies.

This is the smallest possible TelemetryDataset that exercises the full
TelemetryDataset Protocol. Its purpose is solely to smoke-test the scaffold —
for realistic complexity (6-layer composable generator with multiple noise
distributions and labeled fault injection) see `dgx_ts_lab.datasets.synthetic.layered`
in Phase 1.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

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


class TrivialSyntheticDataset:
    """Multivariate sine waves with rare per-step spike anomalies."""

    def __init__(
        self,
        n_samples: int = 10_000,
        n_channels: int = 4,
        sample_rate_hz: float = 1.0,
        noise_std: float = 0.05,
        anomaly_rate: float = 0.01,
        anomaly_magnitude_sigmas: float = 8.0,
        seed: int = 0,
        name: str = "trivial_synth",
        _arrays: dict | None = None,
    ) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._name = name

        if _arrays is not None:
            # Internal: used by split() to create sliced sub-datasets without regenerating.
            self._data = _arrays["data"]
            self._timestamps = _arrays["timestamps"]
            self._labels = _arrays["labels"]
            self._channels = _arrays["channels"]
            return

        rng = np.random.default_rng(seed)
        t = np.arange(n_samples, dtype=np.float32)

        # Per-channel sine at distinct (incommensurate) periods.
        base_periods = np.array([200.0, 137.0, 91.0, 60.0, 43.0, 29.0], dtype=np.float32)
        periods = base_periods[:n_channels]
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_channels).astype(np.float32)
        freqs = 1.0 / periods
        # (T, C) = sin(2π f t + φ)
        signal = np.sin(2.0 * np.pi * np.outer(t, freqs) + phases[None, :]).astype(np.float32)
        noise = rng.normal(0.0, noise_std, size=signal.shape).astype(np.float32)
        data = signal + noise

        # Inject point anomalies: a fraction of timesteps get a large additive spike
        # on a random subset of their channels.
        n_anomalies = max(1, int(n_samples * anomaly_rate))
        anomaly_idx = rng.choice(n_samples, size=n_anomalies, replace=False)
        labels = np.zeros(n_samples, dtype=np.bool_)
        labels[anomaly_idx] = True
        spike_mag = anomaly_magnitude_sigmas * noise_std
        for i in anomaly_idx:
            n_affected = int(rng.integers(1, n_channels + 1))
            chans = rng.choice(n_channels, size=n_affected, replace=False)
            sign = float(rng.choice([-1.0, 1.0]))
            data[i, chans] += sign * spike_mag

        self._data = data
        self._timestamps = (t * (1000.0 / sample_rate_hz)).astype(np.int64)
        self._labels = labels
        self._channels = tuple(
            Channel(
                name=f"ch{i}",
                units=Units.DIMENSIONLESS,
                subsystem=Subsystem.UNKNOWN,
                sample_rate_hz=sample_rate_hz,
                description=f"sine wave channel {i} (period {periods[i]:.0f} samples)",
            )
            for i in range(n_channels)
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def subsystem(self) -> Subsystem:
        return Subsystem.UNKNOWN

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

    def split(self, scheme: SplitScheme) -> Mapping[str, "TrivialSyntheticDataset"]:
        n = len(self._data)
        n_train = int(scheme.train_frac * n)
        n_val = int(scheme.val_frac * n)
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n),
        }
        return {
            split_name: TrivialSyntheticDataset(
                sample_rate_hz=self._sample_rate_hz,
                name=f"{self._name}_{split_name}",
                _arrays={
                    "data": self._data[s:e].copy(),
                    "timestamps": self._timestamps[s:e].copy(),
                    "labels": self._labels[s:e].copy(),
                    "channels": self._channels,
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


@DATASET_REGISTRY.register("trivial_synth")
def _create(**kwargs: object) -> TrivialSyntheticDataset:
    return TrivialSyntheticDataset(**kwargs)  # type: ignore[arg-type]
