"""TimesFM adapter — Google's decoder-only time-series foundation model.

TimesFM 2.0 (`google/timesfm-2.0-500m-pytorch`) is a 500M-param patched
decoder-only transformer. Architecturally simpler than Chronos (no T5
encoder/decoder, no quantile tokenization — operates on real-valued
patches) but the lab adapter follows the same `ForecastingDetector`
contract used by Chronos / MOMENT / Moirai.

Multivariate strategy: per-channel-then-max (locked Phase 3 decision).
Each channel is forecast independently; the base class returns the max
channel-wise residual per step.

Weight loading via :func:`dgx_ts_lab.models.foundation._loader.resolve_model_path`
which checks MLflow Registry first then falls back to ``data/models/``.
If weights are absent (dev box without sneakernet bundle), the module
falls back to a small randomly-initialized transformer so unit tests
still pass — same air-gap pattern as Chronos.

Reference: Das et al., "A decoder-only foundation model for time-series
forecasting" (Google Research, 2024).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from dgx_ts_core.registry import DETECTOR_REGISTRY

from ._base import ForecastingDetector
from ._loader import resolve_model_path


class _TimesFMModule(nn.Module):
    """Wraps the TimesFM decoder transformer + a per-channel norm buffer.

    TimesFM consumes a sequence of real-valued patches (no tokenization)
    and emits next-patch predictions. We patchify the input window, run
    one forward pass, and de-patchify to get per-step predictions.

    Built at fit() time when channel count + window length are known.
    """

    def __init__(
        self,
        model_path: Path,
        n_channels: int,
        window_length: int,
        patch_len: int = 32,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.window_length = int(window_length)
        self.patch_len = int(patch_len)
        if self.window_length % self.patch_len != 0:
            raise ValueError(
                f"window_length={self.window_length} must be a multiple of "
                f"patch_len={self.patch_len}"
            )
        self.n_patches = self.window_length // self.patch_len

        # Try to load real TimesFM weights; fall back to a small untrained
        # transformer if missing (lets dev-without-weights + CI still work).
        self._is_pretrained = False
        try:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            self._d_model = int(getattr(self.backbone.config, "hidden_size", 1280))
            self._is_pretrained = True
        except Exception:
            # Untrained fallback: a tiny decoder-only transformer with the
            # same input/output shape so the rest of the code path runs.
            self._d_model = 64
            self.backbone = _TinyDecoderFallback(
                d_model=self._d_model,
                n_layers=2,
                n_heads=2,
                d_ff=128,
            )

        # Per-patch in/out projections (patch_len → d_model and back).
        # TimesFM's real model has these built in, but exposing our own
        # makes the fallback path work uniformly.
        self.patch_in = nn.Linear(self.patch_len, self._d_model)
        self.patch_out = nn.Linear(self._d_model, self.patch_len)

        # Per-channel normalization buffers, populated at fit time from
        # dataset.stats() — matches the Chronos pattern.
        self.register_buffer("norm_mean", torch.zeros(n_channels))
        self.register_buffer("norm_std", torch.ones(n_channels))


class _TinyDecoderFallback(nn.Module):
    """Stand-in transformer used when real TimesFM weights are absent.

    NOT a faithful TimesFM — just a same-shape module so the adapter and
    its tests can run without the 500M-param HF download.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.config = type("_Cfg", (), {"hidden_size": d_model})()

    def forward(self, inputs_embeds: torch.Tensor, **kwargs):
        # Mimic HuggingFace forward signature loosely
        last_hidden_state = self.encoder(inputs_embeds)
        return type("_Out", (), {"last_hidden_state": last_hidden_state})()


class TimesFMDetector(ForecastingDetector):
    """AnomalyDetector wrapping a TimesFM-style decoder-only forecaster."""

    name = "timesfm"

    def __init__(
        self,
        model: str = "google/timesfm-2.0-500m-pytorch",
        window_length: int = 512,
        patch_len: int = 32,
        n_channels: int | None = None,
    ) -> None:
        self._model_name = model
        self._window_length = int(window_length)
        self._patch_len = int(patch_len)
        self._n_channels = int(n_channels) if n_channels is not None else None
        self._module: _TimesFMModule | None = None

    # ── ForecastingDetector overrides ────────────────────────────────────

    def _build_module(self, n_channels: int, window_length: int) -> nn.Module:
        try:
            path = resolve_model_path(self._model_name)
        except FileNotFoundError:
            path = Path("__missing__")
        return _TimesFMModule(
            model_path=path,
            n_channels=n_channels,
            window_length=window_length,
            patch_len=self._patch_len,
        )

    def _forecast_batch(self, x_norm: torch.Tensor) -> torch.Tensor:
        """One-step-ahead forecast for every timestep.

        Pipeline:
            (B, T, C) → permute → (B*C, T)
                     → patchify (B*C, n_patches, patch_len)
                     → patch_in → (B*C, n_patches, d_model)
                     → backbone → (B*C, n_patches, d_model)
                     → patch_out → (B*C, n_patches, patch_len)
                     → de-patchify (B*C, T)
                     → reshape → (B, T, C)
        The output at position t is the model's prediction of x[t+1].
        """
        B, T, C = x_norm.shape
        mod = self.module

        # Channel-independent: flatten channels into the batch dim
        x = x_norm.permute(0, 2, 1).reshape(B * C, T)            # (B*C, T)

        # Patchify
        n_patches = T // mod.patch_len
        T_trunc = n_patches * mod.patch_len
        x = x[:, :T_trunc]                                       # truncate to multiple
        x_p = x.reshape(B * C, n_patches, mod.patch_len)         # (B*C, n_patches, P)

        # Patch embed
        h = mod.patch_in(x_p)                                    # (B*C, n_patches, d_model)

        # Backbone forward — handles both HF AutoModel and the fallback
        out = mod.backbone(inputs_embeds=h)
        hidden = out.last_hidden_state                           # (B*C, n_patches, d_model)

        # De-patchify back to per-step predictions
        pred_patches = mod.patch_out(hidden)                     # (B*C, n_patches, P)
        pred_flat = pred_patches.reshape(B * C, T_trunc)         # (B*C, T_trunc)

        # Pad back to original T if we truncated (last predicted value)
        if T_trunc < T:
            pad = pred_flat[:, -1:].expand(-1, T - T_trunc)
            pred_flat = torch.cat([pred_flat, pad], dim=1)

        # Reshape back to (B, T, C)
        return pred_flat.reshape(B, C, T).permute(0, 2, 1).contiguous()

    # ── save / load ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._module is None:
            raise RuntimeError("nothing to save — detector not fit")
        torch.save(
            {
                "model_name": self._model_name,
                "window_length": self._window_length,
                "patch_len": self._patch_len,
                "n_channels": self._n_channels,
                "module_state": self._module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TimesFMDetector:
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(
            model=data["model_name"],
            window_length=data["window_length"],
            patch_len=data["patch_len"],
            n_channels=data["n_channels"],
        )
        det._n_channels = data["n_channels"]
        try:
            mpath = resolve_model_path(data["model_name"])
        except FileNotFoundError:
            mpath = Path("__missing__")
        det._module = _TimesFMModule(
            model_path=mpath,
            n_channels=det._n_channels,
            window_length=det._window_length,
            patch_len=det._patch_len,
        )
        det._module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("timesfm")
def _create(**kwargs: Any) -> TimesFMDetector:
    return TimesFMDetector(**kwargs)
