"""PINN wrapper that subtracts an analytical physics prediction from input.

A `PhysicsModel` is a (stateless or stateful) function that takes a
TelemetryWindow and returns a prediction tensor of the same shape, populating
the channels it knows about with model predictions and leaving the rest at
zero. The wrapper computes ``residual = window.tensor − physics_pred`` and
passes a new TelemetryWindow (same metadata, residual tensor) to the inner
detector. Labels and channels are preserved.

This is the **pre-process mode** locked for Phase 4. Other modes (joint
training, post-process re-rank) can be added later as separate wrappers
without touching this one.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from dgx_ts_core.data import (
    Channel,
    DatasetStats,
    SplitScheme,
    Subsystem,
    TelemetryDataset,
    TelemetryWindow,
)
from dgx_ts_core.models import (
    AnomalyDetector,
    AnomalyScore,
    Capabilities,
    FitMode,
    FitResult,
)
from dgx_ts_core.registry import DETECTOR_REGISTRY


@runtime_checkable
class PhysicsModel(Protocol):
    """A physics model that predicts a subset of telemetry channels.

    Implementations should be cheap and side-effect-free. They receive the
    full window (so they can read mode signals, time, etc.) and return a
    prediction array shaped like the window's tensor — zeros for channels
    the model doesn't cover.
    """

    @property
    def name(self) -> str: ...

    def predict(self, window: TelemetryWindow) -> np.ndarray:
        """Return predicted values shaped (T, C) matching window.tensor."""
        ...

    def covered_channels(self) -> set[str]:
        """Names of channels this model predicts. Used for diagnostics."""
        ...


class _ResidualDataset:
    """Wraps a TelemetryDataset, subtracting physics predictions at window time."""

    def __init__(self, inner: TelemetryDataset, physics: PhysicsModel) -> None:
        self._inner = inner
        self._physics = physics
        self._cached_stats: DatasetStats | None = None

    @property
    def name(self) -> str:
        return f"{self._inner.name}_residual_{self._physics.name}"

    @property
    def subsystem(self) -> Subsystem:
        return self._inner.subsystem

    @property
    def channels(self) -> tuple[Channel, ...]:
        return self._inner.channels

    @property
    def sample_rate_hz(self) -> float:
        return self._inner.sample_rate_hz

    @property
    def has_labels(self) -> bool:
        return self._inner.has_labels

    def _residual_window(self, w: TelemetryWindow) -> TelemetryWindow:
        pred = self._physics.predict(w)
        return TelemetryWindow(
            tensor=(w.tensor - pred).astype(np.float32),
            timestamps=w.timestamps,
            channels=w.channels,
            mask=w.mask,
            labels=w.labels,
            provenance={**w.provenance, "physics": self._physics.name},
        )

    def windows(self, length: int, stride: int) -> Iterator[TelemetryWindow]:
        for w in self._inner.windows(length=length, stride=stride):
            yield self._residual_window(w)

    def split(self, scheme: SplitScheme) -> Mapping[str, "_ResidualDataset"]:
        return {
            split_name: _ResidualDataset(sub, self._physics)
            for split_name, sub in self._inner.split(scheme).items()
        }

    def stats(self) -> DatasetStats:
        # Approximate: compute stats on a few residual windows then return.
        if self._cached_stats is not None:
            return self._cached_stats
        chunks: list[np.ndarray] = []
        for w in self.windows(length=512, stride=512):
            chunks.append(w.tensor)
            if len(chunks) * 512 >= 20_000:
                break
        if not chunks:
            return self._inner.stats()
        residual_data = np.concatenate(chunks, axis=0)
        self._cached_stats = DatasetStats(
            means=residual_data.mean(axis=0).astype(np.float32),
            stds=(residual_data.std(axis=0) + 1e-8).astype(np.float32),
            p99=np.quantile(np.abs(residual_data), 0.99, axis=0).astype(np.float32),
            n_samples=int(residual_data.shape[0]),
            n_channels=int(residual_data.shape[1]),
        )
        return self._cached_stats

    # Expose the underlying arrays so WindowTorchDataset can use the fast path.
    @property
    def _data(self) -> np.ndarray:
        if not hasattr(self, "_residual_data_cache"):
            inner_data = self._inner._data
            chunks: list[np.ndarray] = []
            # Walk in window-sized chunks so the physics model sees realistic context.
            chunk_len = 1024
            for start in range(0, len(inner_data), chunk_len):
                end = min(start + chunk_len, len(inner_data))
                # Build a "window" representing this chunk so physics can read mode/time.
                tw = TelemetryWindow(
                    tensor=inner_data[start:end].copy(),
                    timestamps=self._inner._timestamps[start:end].copy(),
                    channels=self._inner.channels,
                    labels=(
                        self._inner._labels[start:end].copy()
                        if self._inner._labels is not None
                        else None
                    ),
                )
                pred = self._physics.predict(tw)
                chunks.append((tw.tensor - pred).astype(np.float32))
            self._residual_data_cache = np.concatenate(chunks, axis=0)
        return self._residual_data_cache

    @property
    def _labels(self) -> np.ndarray | None:
        return getattr(self._inner, "_labels", None)

    @property
    def _timestamps(self) -> np.ndarray:
        return self._inner._timestamps


class PINNResidualDetector:
    """Composes an inner AnomalyDetector with a PhysicsModel.

    At fit() time: builds a residual dataset (data − physics) and fits the
    inner detector on it. At score() time: subtracts physics from the
    window and scores the residual with the inner detector.
    """

    def __init__(
        self,
        inner: AnomalyDetector,
        physics: PhysicsModel,
    ) -> None:
        self._inner = inner
        self._physics = physics

    @property
    def name(self) -> str:
        return f"pinn[{self._physics.name}]_{self._inner.name}"

    @property
    def capabilities(self) -> Capabilities:
        # Pass through inner caps; add a note in metadata at fit time
        return self._inner.capabilities

    # Expose .module so the trainer's Fabric loop sees the inner's module.
    @property
    def module(self):
        return getattr(self._inner, "module", None)

    @module.setter
    def module(self, value) -> None:
        # Trainer may rewrap with LoRA — forward to inner
        if hasattr(self._inner, "module"):
            self._inner.module = value

    def compute_loss(self, batch):
        # Batch comes from the residual dataset already (Fabric called fit
        # with the residual dataset which set up things correctly).
        return self._inner.compute_loss(batch)

    def compute_score_batch(self, batch):
        return self._inner.compute_score_batch(batch)

    def fit(
        self,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: dict[str, Any],
    ) -> FitResult:
        residual_dataset = _ResidualDataset(dataset, self._physics)
        result = self._inner.fit(residual_dataset, mode, config)
        return FitResult(
            detector_name=self.name,
            mode=result.mode,
            final_loss=result.final_loss,
            n_steps=result.n_steps,
            artifacts=result.artifacts,
            metadata={
                **result.metadata,
                "physics_model": self._physics.name,
                "physics_covered_channels": sorted(self._physics.covered_channels()),
            },
        )

    def score(self, window: TelemetryWindow) -> AnomalyScore:
        pred = self._physics.predict(window)
        residual = TelemetryWindow(
            tensor=(window.tensor - pred).astype(np.float32),
            timestamps=window.timestamps,
            channels=window.channels,
            mask=window.mask,
            labels=window.labels,
            provenance={**window.provenance, "physics": self._physics.name},
        )
        return self._inner.score(residual)

    def embed(self, window):
        return self._inner.embed(window)

    def reconstruct(self, window):
        return self._inner.reconstruct(window)

    def save(self, path: Path) -> None:
        self._inner.save(path)

    @classmethod
    def load(cls, path: Path) -> "PINNResidualDetector":
        raise NotImplementedError(
            "PINNResidualDetector.load needs both inner + physics — "
            "construct manually by loading the inner detector and re-attaching "
            "the physics model."
        )


@DETECTOR_REGISTRY.register("pinn_residual")
def _create(*, inner: dict, physics: dict, **kwargs: Any) -> PINNResidualDetector:
    """Hydra factory for PINN-wrapped detectors.

    Expects ``inner`` and ``physics`` to be dicts with their own ``_target_key``
    pointing at registered factories.
    """
    inner_key = inner.pop("_target_key")
    physics_key = physics.pop("_target_key")
    inner_det = DETECTOR_REGISTRY.create(inner_key, **inner)
    # Physics models live in their own factory — we use a small local map.
    from . import _PHYSICS_REGISTRY

    physics_model = _PHYSICS_REGISTRY[physics_key](**physics)
    return PINNResidualDetector(inner=inner_det, physics=physics_model)
