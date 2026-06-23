"""Shared base class for forecasting-style foundation-model adapters.

Foundation models output forecasts, not anomaly scores. The conversion is:

    1. Slide a context window of length C through the input.
    2. Forecast the next H (horizon) steps from each context.
    3. Compare forecast to the observed values.
    4. score(t) = |forecast(t) - actual(t)| / σ_residual_calibration

For multivariate input with a univariate foundation model, we use the
**per-channel-then-max** strategy: forecast each channel independently in
the batch dimension, then return max_c residual as the per-step score.

This file holds the multivariate-handling logic + sliding-window inference.
Concrete subclasses implement ``_forecast(context: Tensor[B, C, T]) -> Tensor[B, C, H]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from dgx_ts_core.data import TelemetryDataset, TelemetryWindow
from dgx_ts_core.models import (
    AnomalyScore,
    Capabilities,
    FitMode,
    FitResult,
    OutputKind,
)


class ForecastingDetector:
    """Mixin providing forecast→score conversion and multivariate handling.

    Concrete adapters (Chronos, MOMENT, Moirai) inherit OR compose this and
    implement ``_forecast_batch(x: Tensor[B, T, C]) -> Tensor[B, T, C]``
    that returns one-step-ahead predictions for every timestep in the
    window. The base then computes the squared residual per step.

    Each adapter also exposes ``module``, ``compute_loss``, and
    ``compute_score_batch`` to satisfy the neural-detector contract used by
    the Fabric loop.
    """

    # ── subclass contract (override these) ───────────────────────────────

    def _build_module(self, n_channels: int, window_length: int) -> nn.Module:
        """Build and return the nn.Module wrapping the foundation model."""
        raise NotImplementedError

    def _forecast_batch(self, x_norm: torch.Tensor) -> torch.Tensor:
        """One-step-ahead forecast for every position in the window.

        Input:  x_norm   (B, T, C) normalized values
        Output: pred     (B, T, C) predicted next-step values (for t=0..T-1,
                                  pred[:, t, :] predicts the value at t+1; the
                                  last column simply shifts so shape matches).
        """
        raise NotImplementedError

    # ── shared infrastructure ────────────────────────────────────────────

    name: str = "foundation_base"
    _module: nn.Module | None = None
    _n_channels: int | None = None
    _window_length: int = 256

    @property
    def module(self) -> nn.Module:
        if self._module is None:
            raise RuntimeError("module not built — call fit() first")
        return self._module

    @module.setter
    def module(self, value: nn.Module) -> None:
        self._module = value

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            requires_pretraining=True,
            supports_streaming=False,
            supports_multivariate=True,   # via per-channel-then-max
            native_context_len=self._window_length,
            output_kind=OutputKind.PER_STEP,
            supports_peft=True,
            supports_export_onnx=False,
            supports_zero_shot=True,
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]                                  # (B, T, C)
        pred = self._forecast_batch(self._normalize(x))
        x_norm = self._normalize(x)
        # one-step prediction loss: pred[:, t] should match x_norm[:, t+1]
        # Shift the target by 1 step; truncate by 1 to align.
        if pred.shape[1] > 1:
            return ((pred[:, :-1] - x_norm[:, 1:]) ** 2).mean()
        return ((pred - x_norm) ** 2).mean()

    def compute_score_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]                                  # (B, T, C)
        self.module.eval()
        with torch.no_grad():
            x_norm = self._normalize(x)
            pred = self._forecast_batch(x_norm)
            # Per-step squared residual, then per-channel-then-max
            if pred.shape[1] > 1:
                resid = (pred[:, :-1] - x_norm[:, 1:]) ** 2     # (B, T-1, C)
                # Pad first step (no prediction available) with 0
                pad = torch.zeros(resid.shape[0], 1, resid.shape[2], device=resid.device)
                resid = torch.cat([pad, resid], dim=1)          # (B, T, C)
            else:
                resid = (pred - x_norm) ** 2
            per_step = resid.max(dim=-1).values                  # (B, T)
        return per_step

    # ── normalization buffers (populated at fit time) ────────────────────

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        m = self.module.norm_mean
        s = self.module.norm_std
        return (x - m) / s

    def _init_norm(self, dataset: TelemetryDataset) -> None:
        stats = dataset.stats()
        with torch.no_grad():
            self.module.norm_mean.copy_(torch.from_numpy(stats.means).float())
            self.module.norm_std.copy_(
                torch.from_numpy(stats.stds).float().clamp_min(1e-6)
            )

    # ── AnomalyDetector Protocol — score / save / load ───────────────────

    def fit(
        self,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: dict[str, Any],
    ) -> FitResult:
        n_channels = len(dataset.channels)
        if self._module is None:
            self._n_channels = n_channels
            self._module = self._build_module(n_channels, self._window_length)
            # Add normalization buffers if the module didn't already.
            if not hasattr(self._module, "norm_mean"):
                self._module.register_buffer("norm_mean", torch.zeros(n_channels))
                self._module.register_buffer("norm_std", torch.ones(n_channels))
        self._init_norm(dataset)
        return FitResult(
            detector_name=self.name,
            mode=mode,
            n_steps=0,
            metadata={
                "n_parameters_total": sum(p.numel() for p in self._module.parameters()),
                "n_parameters_trainable": sum(
                    p.numel() for p in self._module.parameters() if p.requires_grad
                ),
            },
        )

    def score(self, window: TelemetryWindow) -> AnomalyScore:
        device = next(self.module.parameters()).device
        x = torch.from_numpy(window.tensor).float().unsqueeze(0).to(device)
        scores = (
            self.compute_score_batch({"x": x}).squeeze(0).cpu().numpy().astype(np.float32)
        )
        return AnomalyScore(scores=scores)

    def embed(self, window):
        raise NotImplementedError("foundation adapters do not expose embeddings yet")

    def reconstruct(self, window):
        raise NotImplementedError("foundation adapters are forecasting-based")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._module is None:
            raise RuntimeError("nothing to save — detector not fit")
        torch.save(
            {
                "name": self.name,
                "n_channels": self._n_channels,
                "window_length": self._window_length,
                "module_state": self._module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> ForecastingDetector:
        # Subclasses override — base can't reconstruct the module type.
        raise NotImplementedError(
            f"{cls.__name__} must override load() to rebuild the module type"
        )
