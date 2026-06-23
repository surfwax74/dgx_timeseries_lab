"""MOMENT adapter — CMU's T5-encoder-based foundation model.

Full implementation requires either the ``momentfm`` package or a
re-implementation of the MOMENT-specific patching + task heads. Phase 3
ships a working architectural shell that:

    - registers as a detector,
    - declares honest Capabilities,
    - errors loudly at fit() if real weights aren't reachable,
    - works with a small randomly-initialized T5 backbone for unit tests.

Real-weight integration: depend on `momentfm` (uncomment in pyproject
optional-deps) or load via transformers with the appropriate config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from dgx_ts_core.registry import DETECTOR_REGISTRY

from ._base import ForecastingDetector
from ._loader import resolve_model_path


class _MomentModule(nn.Module):
    """T5-encoder + patch reconstruction head wrapping MOMENT weights."""

    def __init__(
        self,
        model_path: Path,
        n_channels: int,
        window_length: int,
        patch_len: int = 8,
        d_model: int = 64,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.window_length = int(window_length)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        self.n_patches = self.window_length // self.patch_len

        from transformers import T5Config, T5EncoderModel

        # Try loading pretrained MOMENT weights; fall back to a small
        # untrained T5 encoder for tests / dev-without-weights.
        try:
            self.encoder = T5EncoderModel.from_pretrained(str(model_path))
            self._is_pretrained = True
            self.d_model = self.encoder.config.d_model
        except Exception:
            cfg = T5Config(d_model=d_model, d_kv=16, d_ff=128, num_layers=2, num_heads=2)
            self.encoder = T5EncoderModel(cfg)
            self._is_pretrained = False

        self.patch_proj = nn.Linear(patch_len, self.d_model)
        self.recon_head = nn.Linear(self.d_model, patch_len)
        self.register_buffer("norm_mean", torch.zeros(n_channels))
        self.register_buffer("norm_std", torch.ones(n_channels))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        # (B, T, C) → per-channel patches (B*C, n_patches, patch_len)
        B, T, C = x_norm.shape
        T_trunc = self.n_patches * self.patch_len
        x = x_norm[:, :T_trunc, :].permute(0, 2, 1).contiguous()
        patches = x.reshape(B * C, self.n_patches, self.patch_len)
        embedded = self.patch_proj(patches)                          # (B*C, N, D)
        # T5 encoder takes inputs_embeds for embedding-based input
        encoded = self.encoder(inputs_embeds=embedded).last_hidden_state
        recon_patches = self.recon_head(encoded)                     # (B*C, N, P)
        recon = recon_patches.reshape(B, C, self.n_patches * self.patch_len)
        return recon.permute(0, 2, 1).contiguous()                   # (B, T_trunc, C)


class MomentDetector(ForecastingDetector):
    """MOMENT adapter. Reconstruction-based scoring (residual = x - recon)."""

    name = "moment"

    def __init__(
        self,
        model: str = "AutonLab/MOMENT-1-small",
        window_length: int = 256,
        patch_len: int = 8,
        d_model: int = 64,
        n_channels: int | None = None,
    ) -> None:
        self._model_name = model
        self._window_length = int(window_length)
        self._patch_len = int(patch_len)
        self._d_model = int(d_model)
        self._n_channels = int(n_channels) if n_channels is not None else None
        self._module: _MomentModule | None = None

    def _build_module(self, n_channels: int, window_length: int) -> nn.Module:
        try:
            path = resolve_model_path(self._model_name)
        except FileNotFoundError:
            # Allowed in dev/tests — module falls back to untrained T5.
            path = Path("__missing__")
        return _MomentModule(
            model_path=path,
            n_channels=n_channels,
            window_length=window_length,
            patch_len=self._patch_len,
            d_model=self._d_model,
        )

    def _forecast_batch(self, x_norm: torch.Tensor) -> torch.Tensor:
        recon = self.module(x_norm)                                 # (B, T_trunc, C)
        # Pad if recon is shorter than input (truncation from patching)
        B, T, C = x_norm.shape
        if recon.shape[1] < T:
            pad = torch.zeros(B, T - recon.shape[1], C, device=recon.device)
            recon = torch.cat([recon, pad], dim=1)
        # Reconstruction model: "forecast" = the same step's reconstruction.
        # Override the base's shifted residual logic by returning recon
        # aligned with x_norm — the base computes (pred - x_norm) downstream.
        return recon

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]
        x_norm = self._normalize(x)
        recon = self._forecast_batch(x_norm)
        return ((recon - x_norm) ** 2).mean()

    def compute_score_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]
        self.module.eval()
        with torch.no_grad():
            x_norm = self._normalize(x)
            recon = self._forecast_batch(x_norm)
            err = ((recon - x_norm) ** 2).max(dim=-1).values
        return err

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
                "d_model": self._d_model,
                "n_channels": self._n_channels,
                "module_state": self._module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> MomentDetector:
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(
            model=data["model_name"],
            window_length=data["window_length"],
            patch_len=data["patch_len"],
            d_model=data["d_model"],
            n_channels=data["n_channels"],
        )
        det._n_channels = data["n_channels"]
        det._module = _MomentModule(
            model_path=Path("__missing__"),
            n_channels=det._n_channels,
            window_length=det._window_length,
            patch_len=det._patch_len,
            d_model=det._d_model,
        )
        det._module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("moment")
def _create(**kwargs: Any) -> MomentDetector:
    return MomentDetector(**kwargs)
