"""Sat-TSFM — channel-flexible satellite-telemetry transformer.

The Phase 4 from-scratch foundation model. Designed to scale from ~5M
(dev tier) to 1B+ params (DGX 8×H200 FSDP) with the same architecture.

Key design choices (locked Phase 4):

    1. **Continuous-patch tokenization** (Moirai-style): each channel's
       window is split into non-overlapping patches of length P; each
       patch is linearly embedded to d_model.

    2. **Learnable channel-position embeddings**: a lookup table indexed
       by channel id (0..max_channels-1). Datasets pass their channel
       indices; same model serves any C ≤ max_channels.

    3. **Joint time + channel attention**: patches reshaped to
       (B, C·N, D) so a single transformer encoder attends across both
       channels and time. Channel cross-talk is learned, not hard-coded.

    4. **Per-channel forecast head**: predicts the next patch per channel
       (B*C, N+1, P). At inference, residual = predicted − actual gives
       per-step anomaly scores.

Size variants (params count is approximate):

    tiny    d_model=64   layers=2  heads=2     ~1M    CPU/RTX 3080
    small   d_model=128  layers=4  heads=4    ~10M    RTX 3080 / A5000
    medium  d_model=256  layers=8  heads=8   ~100M    A5000 / A5000×8
    large   d_model=512  layers=12 heads=16  ~500M    A5000×8 FSDP / H200
    xlarge  d_model=1024 layers=24 heads=32   ~1B+    8×H200 FSDP
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


class SatTSFMModule(nn.Module):
    """Channel-flexible transformer with patch + channel + time embeddings."""

    def __init__(
        self,
        max_channels: int = 256,
        window_length: int = 256,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if window_length % patch_len != 0:
            raise ValueError(
                f"window_length={window_length} must be a multiple of patch_len={patch_len}"
            )
        self.max_channels = int(max_channels)
        self.window_length = int(window_length)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        self.n_patches = self.window_length // self.patch_len

        # Patch embedding (continuous tokenization — Moirai-style)
        self.patch_embed = nn.Linear(self.patch_len, self.d_model)
        # Channel-position embedding (variable C up to max_channels)
        self.channel_emb = nn.Embedding(self.max_channels, self.d_model)
        # Time-position embedding (fixed N patches per channel)
        self.time_emb = nn.Parameter(torch.zeros(1, self.n_patches, self.d_model))
        nn.init.trunc_normal_(self.time_emb, std=0.02)
        nn.init.trunc_normal_(self.channel_emb.weight, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.forecast_head = nn.Linear(self.d_model, self.patch_len)

        # Per-channel normalization (filled at fit time)
        self.register_buffer("norm_mean", torch.zeros(self.max_channels))
        self.register_buffer("norm_std", torch.ones(self.max_channels))

    def encode(
        self, x: torch.Tensor, channel_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int, int, int]:
        """Run the encoder up to (but not including) the forecast head.

        Returns ``(encoded, B, C, n_patches)`` where ``encoded`` has shape
        ``(B, C*N, D)``. Used by ``forward`` for reconstruction AND by
        multi-task task heads that need per-step embeddings.
        """
        B, T, C = x.shape
        if channel_ids is None:
            channel_ids = torch.arange(C, device=x.device, dtype=torch.long)
        mean = self.norm_mean[channel_ids].view(1, 1, C)
        std = self.norm_std[channel_ids].view(1, 1, C)
        x_norm = (x - mean) / std

        T_trunc = self.n_patches * self.patch_len
        x_p = x_norm[:, :T_trunc, :].permute(0, 2, 1).contiguous()
        patches = x_p.reshape(B, C, self.n_patches, self.patch_len)

        emb = self.patch_embed(patches)
        emb = emb + self.time_emb.unsqueeze(1)
        ch_emb = self.channel_emb(channel_ids).view(1, C, 1, self.d_model)
        emb = emb + ch_emb
        flat = emb.reshape(B, C * self.n_patches, self.d_model)
        encoded = self.encoder(flat)
        return encoded, B, C, self.n_patches

    def encode_pooled_steps(
        self, x: torch.Tensor, channel_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-step channel-pooled embeddings for multi-task heads.

        Returns ``(B, T_trunc, D)`` where ``T_trunc = n_patches *
        patch_len``. Each patch's embedding is broadcast to its
        ``patch_len`` constituent timesteps; channel dimension is
        mean-pooled. This is the input multi-task heads consume.
        """
        encoded, B, C, N = self.encode(x, channel_ids)         # (B, C*N, D)
        # (B, C, N, D) → mean over C → (B, N, D)
        per_patch = encoded.reshape(B, C, N, self.d_model).mean(dim=1)
        # Broadcast each patch to its patch_len steps → (B, N, patch_len, D) → (B, T_trunc, D)
        per_step = per_patch.unsqueeze(2).expand(B, N, self.patch_len, self.d_model)
        return per_step.reshape(B, N * self.patch_len, self.d_model)

    def forward(
        self, x: torch.Tensor, channel_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: (B, T, C); returns reconstruction (B, T, C) in normalized space.

        ``channel_ids``: (C,) long tensor mapping dataset channels to
        embedding-table indices. Defaults to identity if not provided.
        """
        T = x.shape[1]
        encoded, B, C, N = self.encode(x, channel_ids)
        T_trunc = N * self.patch_len
        recon_patches = self.forecast_head(encoded)       # (B, C*N, P)
        # Reshape back to (B, C, N, P) then to (B, T, C)
        recon = recon_patches.reshape(B, C, N, self.patch_len)
        recon = recon.reshape(B, C, T_trunc).permute(0, 2, 1).contiguous()

        # Pad to original T if truncated
        if recon.shape[1] < T:
            pad = torch.zeros(B, T - recon.shape[1], C, device=recon.device)
            recon = torch.cat([recon, pad], dim=1)
        return recon


class SatTSFMDetector:
    """AnomalyDetector wrapping SatTSFMModule. Reconstruction-based scoring."""

    def __init__(
        self,
        window_length: int = 256,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_channels: int = 256,
        n_channels: int | None = None,
    ) -> None:
        self._cfg: dict[str, Any] = dict(
            window_length=int(window_length),
            patch_len=int(patch_len),
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            d_ff=int(d_ff),
            dropout=float(dropout),
            max_channels=int(max_channels),
        )
        self._n_channels: int | None = int(n_channels) if n_channels is not None else None
        self.module: SatTSFMModule | None = None

    @property
    def name(self) -> str:
        return "sat_tsfm"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            requires_pretraining=True,
            supports_streaming=False,
            supports_multivariate=True,
            native_context_len=self._cfg["window_length"],
            output_kind=OutputKind.PER_STEP,
            supports_peft=True,
            supports_export_onnx=True,
            supports_export_threshold_baked=True,
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        recon = self.module(x)
        # Normalize x to match recon space
        C = x.shape[-1]
        ids = torch.arange(C, device=x.device, dtype=torch.long)
        m = self.module.norm_mean[ids].view(1, 1, C)
        s = self.module.norm_std[ids].view(1, 1, C)
        x_norm = (x - m) / s
        return ((recon - x_norm) ** 2).mean()

    def compute_score_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        self.module.eval()
        with torch.no_grad():
            recon = self.module(x)
            C = x.shape[-1]
            ids = torch.arange(C, device=x.device, dtype=torch.long)
            m = self.module.norm_mean[ids].view(1, 1, C)
            s = self.module.norm_std[ids].view(1, 1, C)
            x_norm = (x - m) / s
            err = ((recon - x_norm) ** 2).max(dim=-1).values
        return err

    def fit(
        self,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: dict[str, Any],
    ) -> FitResult:
        n_channels = len(dataset.channels)
        if self._n_channels is None:
            self._n_channels = n_channels
        if self.module is None:
            self.module = SatTSFMModule(**self._cfg)
        if n_channels > self.module.max_channels:
            raise ValueError(
                f"dataset has {n_channels} channels but max_channels={self.module.max_channels}; "
                "increase max_channels in the model config."
            )
        # Populate normalization buffers for the channels we'll see.
        stats = dataset.stats()
        with torch.no_grad():
            self.module.norm_mean[:n_channels].copy_(
                torch.from_numpy(stats.means).float()
            )
            self.module.norm_std[:n_channels].copy_(
                torch.from_numpy(stats.stds).float().clamp_min(1e-6)
            )
        n_params = sum(p.numel() for p in self.module.parameters())
        return FitResult(
            detector_name=self.name,
            mode=mode,
            n_steps=0,
            metadata={
                "n_parameters": n_params,
                "n_parameters_human": _human_params(n_params),
                "n_channels_seen": n_channels,
                "max_channels": self.module.max_channels,
            },
        )

    def score(self, window: TelemetryWindow) -> AnomalyScore:
        if self.module is None:
            raise RuntimeError("must fit before scoring")
        device = next(self.module.parameters()).device
        x = torch.from_numpy(window.tensor).float().unsqueeze(0).to(device)
        scores = (
            self.compute_score_batch({"x": x}).squeeze(0).cpu().numpy().astype(np.float32)
        )
        return AnomalyScore(scores=scores)

    def embed(self, window):
        raise NotImplementedError

    def reconstruct(self, window):
        if self.module is None:
            raise RuntimeError("must fit before reconstructing")
        device = next(self.module.parameters()).device
        x = torch.from_numpy(window.tensor).float().unsqueeze(0).to(device)
        self.module.eval()
        with torch.no_grad():
            recon = self.module(x)
            C = x.shape[-1]
            ids = torch.arange(C, device=device, dtype=torch.long)
            m = self.module.norm_mean[ids].view(1, 1, C)
            s = self.module.norm_std[ids].view(1, 1, C)
            recon_real = recon * s + m
        return recon_real.squeeze(0).cpu().numpy()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.module is None:
            raise RuntimeError("nothing to save — detector not fit")
        torch.save(
            {
                "config": self._cfg,
                "n_channels": self._n_channels,
                "module_state": self.module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SatTSFMDetector":
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(n_channels=data["n_channels"], **data["config"])
        det._n_channels = data["n_channels"]
        det.module = SatTSFMModule(**det._cfg)
        det.module.load_state_dict(data["module_state"])
        return det


def _human_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    if n >= 1_000:
        return f"{n/1e3:.0f}K"
    return str(n)


@DETECTOR_REGISTRY.register("sat_tsfm")
def _create(**kwargs: Any) -> SatTSFMDetector:
    return SatTSFMDetector(**kwargs)
