"""TTM adapter — IBM Tiny Time Mixer.

`ibm-granite/granite-timeseries-ttm-r2` (and r1) is an MLP-Mixer-based
time-series foundation model. Tiny by foundation-model standards (1-5M
params depending on context/horizon variant) and explicitly designed for
fast LoRA fine-tuning on commodity GPUs.

**Architecturally distinct from the other adapters**: TTM is NOT a
transformer. It's a stack of MLP-Mixer blocks operating on patched
time-series tokens. We still expose it through the same
`ForecastingDetector` contract — the adapter doesn't care what's inside
the backbone, only that it consumes (B, T, C) and produces a forecast of
the same shape.

Hugging Face exposes TTM via the `tsfm_public` package and the standard
`AutoModel` path. We use `AutoModel` with `trust_remote_code=True` and
fall back to a tiny MLP-Mixer-shaped stand-in when weights are absent —
same air-gap pattern as Chronos / TimesFM.

Reference: Ekambaram et al., "Tiny Time Mixers (TTMs): Fast Pre-trained
Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time
Series" (IBM Research, 2024).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from dgx_ts_core.registry import DETECTOR_REGISTRY

from ._base import ForecastingDetector
from ._loader import resolve_model_path


class _TTMModule(nn.Module):
    """Wraps the TTM MLP-Mixer backbone + per-channel norm buffer.

    TTM expects patched real-valued inputs. We patchify the input window,
    run one forward pass through the mixer stack, and de-patchify to per-
    step predictions.
    """

    def __init__(
        self,
        model_path: Path,
        n_channels: int,
        window_length: int,
        patch_len: int = 16,
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

        self._is_pretrained = False
        try:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            # TTM hidden size varies by variant; safe attribute fallback chain.
            self._d_model = int(
                getattr(self.backbone.config, "d_model", None)
                or getattr(self.backbone.config, "hidden_size", 192)
            )
            self._is_pretrained = True
        except Exception:
            # Untrained fallback: tiny MLP-Mixer-style stack with the same
            # input/output shape. Not a faithful TTM — just enough that
            # the rest of the adapter + tests run without HF weights.
            self._d_model = 64
            self.backbone = _TinyMixerFallback(
                d_model=self._d_model,
                n_patches=self.n_patches,
                n_blocks=2,
            )

        self.patch_in = nn.Linear(self.patch_len, self._d_model)
        self.patch_out = nn.Linear(self._d_model, self.patch_len)

        self.register_buffer("norm_mean", torch.zeros(n_channels))
        self.register_buffer("norm_std", torch.ones(n_channels))


class _TinyMixerFallback(nn.Module):
    """Stand-in MLP-Mixer used when real TTM weights are absent.

    Mixes across the patch dimension (token mixing) and across the
    feature dimension (channel mixing). Two-block default.
    """

    def __init__(self, d_model: int, n_patches: int, n_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_MixerBlock(d_model=d_model, n_patches=n_patches) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.config = type("_Cfg", (), {"hidden_size": d_model, "d_model": d_model})()

    def forward(self, inputs_embeds: torch.Tensor, **kwargs):
        h = inputs_embeds
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return type("_Out", (), {"last_hidden_state": h})()


class _MixerBlock(nn.Module):
    """One MLP-Mixer block: token-mix then channel-mix, both with residuals."""

    def __init__(self, d_model: int, n_patches: int) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(d_model)
        self.token_mix = nn.Sequential(
            nn.Linear(n_patches, n_patches * 2),
            nn.GELU(),
            nn.Linear(n_patches * 2, n_patches),
        )
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mix = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, n_patches, d_model)
        # Token mixing — operates over n_patches dim
        h = self.token_norm(x).transpose(1, 2)            # (B, d_model, n_patches)
        h = self.token_mix(h).transpose(1, 2)             # (B, n_patches, d_model)
        x = x + h
        # Channel mixing — operates over d_model dim
        h = self.channel_norm(x)
        h = self.channel_mix(h)
        return x + h


class TTMDetector(ForecastingDetector):
    """AnomalyDetector wrapping IBM TTM (MLP-Mixer-based forecaster)."""

    name = "ttm"

    def __init__(
        self,
        model: str = "ibm-granite/granite-timeseries-ttm-r2",
        window_length: int = 512,
        patch_len: int = 16,
        n_channels: int | None = None,
    ) -> None:
        self._model_name = model
        self._window_length = int(window_length)
        self._patch_len = int(patch_len)
        self._n_channels = int(n_channels) if n_channels is not None else None
        self._module: _TTMModule | None = None

    # ── ForecastingDetector overrides ────────────────────────────────────

    def _build_module(self, n_channels: int, window_length: int) -> nn.Module:
        try:
            path = resolve_model_path(self._model_name)
        except FileNotFoundError:
            path = Path("__missing__")
        return _TTMModule(
            model_path=path,
            n_channels=n_channels,
            window_length=window_length,
            patch_len=self._patch_len,
        )

    def _forecast_batch(self, x_norm: torch.Tensor) -> torch.Tensor:
        """One-step-ahead forecast — same pipeline as TimesFM but the
        backbone happens to be MLP-Mixer instead of a transformer.

        Pipeline:
            (B, T, C) → permute + flatten → (B*C, T)
                     → patchify (B*C, n_patches, patch_len)
                     → patch_in → (B*C, n_patches, d_model)
                     → backbone (mixer blocks) → (B*C, n_patches, d_model)
                     → patch_out → (B*C, n_patches, patch_len)
                     → de-patchify → (B*C, T)
                     → reshape → (B, T, C)
        """
        B, T, C = x_norm.shape
        mod = self.module

        x = x_norm.permute(0, 2, 1).reshape(B * C, T)
        n_patches = T // mod.patch_len
        T_trunc = n_patches * mod.patch_len
        x = x[:, :T_trunc]
        x_p = x.reshape(B * C, n_patches, mod.patch_len)

        h = mod.patch_in(x_p)
        out = mod.backbone(inputs_embeds=h)
        hidden = out.last_hidden_state

        pred_patches = mod.patch_out(hidden)
        pred_flat = pred_patches.reshape(B * C, T_trunc)

        if T_trunc < T:
            pad = pred_flat[:, -1:].expand(-1, T - T_trunc)
            pred_flat = torch.cat([pred_flat, pad], dim=1)

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
    def load(cls, path: Path) -> TTMDetector:
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
        det._module = _TTMModule(
            model_path=mpath,
            n_channels=det._n_channels,
            window_length=det._window_length,
            patch_len=det._patch_len,
        )
        det._module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("ttm")
def _create(**kwargs: Any) -> TTMDetector:
    return TTMDetector(**kwargs)
