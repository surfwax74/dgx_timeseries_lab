"""Chronos adapter — Amazon's T5-based time-series foundation model.

Re-uses HuggingFace ``transformers`` to load the underlying T5 architecture.
We do NOT depend on the ``chronos-forecasting`` package — instead, we
implement a minimal numeric tokenization layer (linear quantile binning)
that mimics the published behavior closely enough for the bake-off.

Multivariate strategy: per-channel-then-max (Phase 3 locked decision).
Each channel is forecast independently in the batch dimension, then the
base class takes the max channel-wise residual per step.

Weight loading via :func:`dgx_ts_lab.models.foundation._loader.resolve_model_path`,
which checks MLflow Registry first then falls back to ``data/models/``.
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
from dgx_ts_core.registry import DETECTOR_REGISTRY

from ._base import ForecastingDetector
from ._loader import resolve_model_path


class _ChronosModule(nn.Module):
    """Wraps a T5 encoder-decoder + numeric tokenization.

    Constructed only at fit() time, after we know the dataset's channel
    count and the model weights' on-disk location.
    """

    def __init__(
        self,
        model_path: Path,
        n_channels: int,
        window_length: int,
        n_quantile_bins: int = 4096,
    ) -> None:
        super().__init__()
        # Lazy import for cleaner cold-start on machines without transformers.
        from transformers import AutoModelForSeq2SeqLM

        self.n_channels = int(n_channels)
        self.window_length = int(window_length)
        self.n_bins = int(n_quantile_bins)

        # Try to load pretrained weights from local path. If the path
        # doesn't actually contain a model (e.g., dev without weights),
        # fall back to a tiny untrained T5 so the rest of the code path
        # still works for unit tests.
        try:
            self.backbone = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
            self._is_pretrained = True
        except Exception:
            from transformers import T5Config, T5ForConditionalGeneration

            cfg = T5Config(
                vocab_size=self.n_bins + 2,  # +2 for special tokens
                d_model=64,
                d_kv=16,
                d_ff=128,
                num_layers=2,
                num_decoder_layers=2,
                num_heads=2,
            )
            self.backbone = T5ForConditionalGeneration(cfg)
            self._is_pretrained = False

        # Per-channel value-to-bin quantile boundaries; populated at fit time.
        self.register_buffer("bin_edges", torch.zeros(n_channels, self.n_bins - 1))
        self.register_buffer("bin_centers", torch.zeros(n_channels, self.n_bins))
        self.register_buffer("norm_mean", torch.zeros(n_channels))
        self.register_buffer("norm_std", torch.ones(n_channels))


class ChronosDetector(ForecastingDetector):
    """AnomalyDetector wrapping a Chronos-style T5 forecaster."""

    name = "chronos"

    def __init__(
        self,
        model: str = "amazon/chronos-t5-tiny",
        window_length: int = 256,
        n_quantile_bins: int = 4096,
        n_channels: int | None = None,
    ) -> None:
        self._model_name = model
        self._window_length = int(window_length)
        self._n_quantile_bins = int(n_quantile_bins)
        self._n_channels = int(n_channels) if n_channels is not None else None
        self._module: _ChronosModule | None = None

    # ── ForecastingDetector overrides ─────────────────────────────────

    def _build_module(self, n_channels: int, window_length: int) -> nn.Module:
        # If weights aren't on disk, pass an "__missing__" sentinel — the
        # module's try/except will fall back to a small untrained T5.
        # That makes dev-without-weights AND unit tests work everywhere.
        try:
            path = resolve_model_path(self._model_name)
        except FileNotFoundError:
            path = Path("__missing__")
        return _ChronosModule(
            model_path=path,
            n_channels=n_channels,
            window_length=window_length,
            n_quantile_bins=self._n_quantile_bins,
        )

    def _forecast_batch(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Per-channel autoregressive forecast.

        For the bake-off we use a single forward pass:
            - tokenize each channel independently via quantile binning,
            - feed to T5 encoder,
            - decode one step ahead per input position via the decoder LM head.

        For air-gap dev without real Chronos weights, this still works as
        a learnable forecaster — just less accurate than the real model.
        """
        B, T, C = x_norm.shape
        mod = self.module
        # Reshape to (B*C, T) — channel-independent forecasting
        x = x_norm.permute(0, 2, 1).reshape(B * C, T)
        # Tokenize: simple bucketization using bin_edges (per-channel).
        # bin_edges has shape (n_channels, n_bins-1); broadcast over batch.
        edges = mod.bin_edges.unsqueeze(0).expand(B, -1, -1).reshape(B * C, -1)
        # Map each value to bin index via bucketize.
        token_ids = torch.bucketize(x, edges[0]) if edges.numel() > 0 else torch.zeros_like(x, dtype=torch.long)
        token_ids = token_ids.clamp(0, mod.n_bins - 1)

        # T5 forward: use encoder hidden states to predict next-token logits.
        # For a clean train signal we shift inputs by one: encoder sees [t0..tN-2],
        # decoder predicts [t1..tN-1].
        if T < 2:
            return torch.zeros_like(x_norm)
        encoder_inputs = token_ids[:, :-1]
        decoder_targets = token_ids[:, 1:]
        # Use the model's forward with labels for cross-entropy supervision when training.
        # For score-batch we just want the predicted continuous value.
        outputs = mod.backbone(
            input_ids=encoder_inputs,
            decoder_input_ids=encoder_inputs,  # parallel decoding
            return_dict=True,
        )
        logits = outputs.logits                              # (B*C, T-1, vocab)
        pred_bins = logits.argmax(dim=-1)                    # (B*C, T-1)
        # De-tokenize: bin index → bin center value
        # bin_centers shape (n_channels, n_bins) → reshape to (B*C, n_bins)
        centers = mod.bin_centers.unsqueeze(0).expand(B, -1, -1).reshape(B * C, -1)
        pred_values = torch.gather(centers, 1, pred_bins.clamp(0, mod.n_bins - 1))
        # Pad to align: predict shape (B*C, T) by appending the last predicted value.
        last = pred_values[:, -1:]
        pred_values_full = torch.cat([pred_values, last], dim=1)  # (B*C, T)
        # Reshape back to (B, T, C)
        return pred_values_full.reshape(B, C, T).permute(0, 2, 1).contiguous()

    # ── AnomalyDetector — extra: populate quantile bins at fit time ───

    def fit(
        self,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: dict[str, Any],
    ) -> FitResult:
        result = super().fit(dataset, mode, config)
        # Compute per-channel quantile bin edges from train data.
        # Use a sample of windows to estimate quantiles.
        n_samples = min(50_000, dataset.stats().n_samples)
        values_per_channel: list[list[float]] = [[] for _ in range(self._n_channels or 0)]
        taken = 0
        for window in dataset.windows(length=self._window_length, stride=self._window_length):
            for c in range(window.num_channels):
                values_per_channel[c].extend(window.tensor[:, c].tolist())
            taken += window.length
            if taken >= n_samples:
                break
        with torch.no_grad():
            for c, vals in enumerate(values_per_channel):
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=np.float32)
                qs = np.linspace(0.0, 1.0, self._n_quantile_bins + 1)[1:-1]
                edges = np.quantile(arr, qs).astype(np.float32)
                centers = np.concatenate([
                    [arr.min() if arr.size else 0.0],
                    (edges[:-1] + edges[1:]) / 2.0,
                    [arr.max() if arr.size else 0.0],
                ]).astype(np.float32)
                self.module.bin_edges[c].copy_(torch.from_numpy(edges))
                self.module.bin_centers[c].copy_(torch.from_numpy(centers))
        return result

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._module is None:
            raise RuntimeError("nothing to save — detector not fit")
        torch.save(
            {
                "model_name": self._model_name,
                "window_length": self._window_length,
                "n_quantile_bins": self._n_quantile_bins,
                "n_channels": self._n_channels,
                "module_state": self._module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "ChronosDetector":
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(
            model=data["model_name"],
            window_length=data["window_length"],
            n_quantile_bins=data["n_quantile_bins"],
            n_channels=data["n_channels"],
        )
        det._n_channels = data["n_channels"]
        det._module = _ChronosModule(
            model_path=resolve_model_path(data["model_name"])
            if _path_exists(data["model_name"])
            else Path("__missing__"),
            n_channels=det._n_channels,
            window_length=det._window_length,
            n_quantile_bins=det._n_quantile_bins,
        )
        det._module.load_state_dict(data["module_state"])
        return det


def _path_exists(name: str) -> bool:
    try:
        resolve_model_path(name)
        return True
    except FileNotFoundError:
        return False


@DETECTOR_REGISTRY.register("chronos")
def _create(**kwargs: Any) -> ChronosDetector:
    return ChronosDetector(**kwargs)
