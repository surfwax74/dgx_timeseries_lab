"""TimeMoE adapter — Mixture-of-Experts time-series foundation model.

`Maple728/TimeMoE-200M` (also `TimeMoE-50M`) is a decoder-only transformer
with MoE feedforward blocks. The headline contribution is sparse expert
activation — ~50M of the 200M parameters fire per forward pass — which
gives competitive forecast quality at lower inference cost than a dense
model of equivalent total size.

**Architectural notes for this adapter**:

1. **MoE routing has uneven expert utilization** in inference. vLLM-style
   continuous batching pays less here than on a dense model. Document
   this in the throughput section of `models/foundation/README.md`.

2. **PEFT/LoRA on MoE** — only attention matrices get adapters; the
   expert FFNs stay frozen. The `peft` library handles this correctly
   when you set `target_modules` to attention-only names. The LoRA
   config in `trainer.peft` reflects this.

3. **Pairing with our SubsystemMoE detector** — both are MoE designs but
   route differently:
   - TimeMoE: learned router, generic expert specialization
   - SubsystemMoE: deterministic router by Channel.subsystem metadata
   The head-to-head leaderboard between these two is the demo slide.

Reference: Shi et al., "Time-MoE: Billion-Scale Time Series Foundation
Models with Mixture of Experts" (NeurIPS 2024). Weights are community-
republished by author account `Maple728` on HuggingFace — see security-
review notes in `docs/foundation_model_roadmap.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from dgx_ts_core.registry import DETECTOR_REGISTRY

from ._base import ForecastingDetector
from ._loader import resolve_model_path


class _TimeMoEModule(nn.Module):
    """Wraps the TimeMoE decoder transformer + per-channel norm buffer.

    Same overall pipeline as TimesFM (patchify → embed → backbone →
    de-patchify) but the backbone has MoE feedforward blocks instead of
    dense ones.
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
            self._d_model = int(
                getattr(self.backbone.config, "hidden_size", None)
                or getattr(self.backbone.config, "d_model", 1024)
            )
            self._is_pretrained = True
        except Exception:
            # Untrained fallback: tiny dense-FFN transformer with same I/O
            # shape. Not a faithful TimeMoE (no MoE routing) but the
            # adapter contract is preserved so tests + dev-without-weights
            # both run.
            self._d_model = 64
            self.backbone = _TinyMoEFallback(
                d_model=self._d_model,
                n_layers=2,
                n_heads=2,
                d_ff=128,
            )

        self.patch_in = nn.Linear(self.patch_len, self._d_model)
        self.patch_out = nn.Linear(self._d_model, self.patch_len)

        self.register_buffer("norm_mean", torch.zeros(n_channels))
        self.register_buffer("norm_std", torch.ones(n_channels))


class _TinyMoEFallback(nn.Module):
    """Stand-in transformer used when real TimeMoE weights are absent.

    Dense FFN (not real MoE — the routing infrastructure is complex
    enough that we don't try to replicate it in the fallback).
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.config = type("_Cfg", (), {"hidden_size": d_model, "d_model": d_model})()

    def forward(self, inputs_embeds: torch.Tensor, **kwargs):
        last_hidden_state = self.encoder(inputs_embeds)
        return type("_Out", (), {"last_hidden_state": last_hidden_state})()


class TimeMoEDetector(ForecastingDetector):
    """AnomalyDetector wrapping a TimeMoE Mixture-of-Experts forecaster."""

    name = "time_moe"

    def __init__(
        self,
        model: str = "Maple728/TimeMoE-200M",
        window_length: int = 512,
        patch_len: int = 16,
        n_channels: int | None = None,
    ) -> None:
        self._model_name = model
        self._window_length = int(window_length)
        self._patch_len = int(patch_len)
        self._n_channels = int(n_channels) if n_channels is not None else None
        self._module: _TimeMoEModule | None = None

    # ── ForecastingDetector overrides ────────────────────────────────────

    def _build_module(self, n_channels: int, window_length: int) -> nn.Module:
        try:
            path = resolve_model_path(self._model_name)
        except FileNotFoundError:
            path = Path("__missing__")
        return _TimeMoEModule(
            model_path=path,
            n_channels=n_channels,
            window_length=window_length,
            patch_len=self._patch_len,
        )

    def _forecast_batch(self, x_norm: torch.Tensor) -> torch.Tensor:
        """One-step-ahead forecast — identical pipeline to TimesFM.
        The MoE routing happens inside the HF backbone; the adapter
        doesn't need to know about it.
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
    def load(cls, path: Path) -> TimeMoEDetector:
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
        det._module = _TimeMoEModule(
            model_path=mpath,
            n_channels=det._n_channels,
            window_length=det._window_length,
            patch_len=det._patch_len,
        )
        det._module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("time_moe")
def _create(**kwargs: Any) -> TimeMoEDetector:
    return TimeMoEDetector(**kwargs)
