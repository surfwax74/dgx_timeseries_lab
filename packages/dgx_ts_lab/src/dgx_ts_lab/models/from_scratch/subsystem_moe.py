"""Subsystem MoE detector — shared backbone + per-subsystem expert heads.

Routes each input channel to its subsystem's expert based on
`Channel.subsystem` metadata. Hard routing (locked Phase 4 decision):
no learned gates, no top-k — the channel's subsystem string IS the routing
decision. This is interpretable and matches our domain ontology.

Architecture:

    1. Shared transformer trunk (PatchTST-style, channel-independent).
    2. Per-subsystem expert MLP head that maps the trunk's per-channel
       hidden state to a reconstruction.
    3. Each channel routes to exactly one expert based on its declared
       Subsystem enum.

Subsystems with no channels in the dataset get their expert weights
preserved but unused (no compute cost per batch — we skip them).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from dgx_ts_core.data import (
    Channel,
    Subsystem,
    TelemetryDataset,
    TelemetryWindow,
)
from dgx_ts_core.models import (
    AnomalyScore,
    Capabilities,
    FitMode,
    FitResult,
    OutputKind,
)
from dgx_ts_core.registry import DETECTOR_REGISTRY


class _SubsystemExpert(nn.Module):
    """Tiny MLP that converts a (B*C, N, D) trunk encoding to (B*C, N, P)."""

    def __init__(self, d_model: int, patch_len: int, hidden: int = None) -> None:
        super().__init__()
        h = hidden or 2 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, patch_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SubsystemMoEModule(nn.Module):
    """Shared trunk + per-subsystem expert heads."""

    def __init__(
        self,
        n_channels: int,
        window_length: int,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        expert_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if window_length % patch_len != 0:
            raise ValueError(
                f"window_length={window_length} must be a multiple of patch_len={patch_len}"
            )
        self.n_channels = int(n_channels)
        self.window_length = int(window_length)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        self.n_patches = self.window_length // self.patch_len

        # Shared trunk
        self.embed = nn.Linear(self.patch_len, self.d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, self.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=int(n_heads), dim_feedforward=int(d_ff),
            dropout=float(dropout), batch_first=True, norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=int(n_layers))

        # Per-subsystem experts — one for every Subsystem enum member.
        self.experts = nn.ModuleDict(
            {
                sub.value: _SubsystemExpert(self.d_model, self.patch_len, expert_hidden)
                for sub in Subsystem
            }
        )

        # Routing — populated at fit() time once we see the dataset's channels.
        self.register_buffer("channel_to_subsystem_id", torch.zeros(n_channels, dtype=torch.long))
        self.register_buffer("norm_mean", torch.zeros(n_channels))
        self.register_buffer("norm_std", torch.ones(n_channels))
        self._subsystem_index = {sub.value: i for i, sub in enumerate(Subsystem)}
        self._subsystem_id_to_name = {i: sub.value for i, sub in enumerate(Subsystem)}

    def set_routing(self, channels: tuple[Channel, ...]) -> None:
        ids = torch.tensor(
            [self._subsystem_index[ch.subsystem.value] for ch in channels],
            dtype=torch.long,
        )
        with torch.no_grad():
            if len(ids) != self.channel_to_subsystem_id.shape[0]:
                self.channel_to_subsystem_id = torch.zeros(len(ids), dtype=torch.long)
            self.channel_to_subsystem_id.copy_(ids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C). Returns reconstruction (B, T, C) in normalized space."""
        B, T, C = x.shape
        T_trunc = self.n_patches * self.patch_len
        x_norm = (x - self.norm_mean) / self.norm_std
        x_p = x_norm[:, :T_trunc, :].permute(0, 2, 1).contiguous()
        patches = x_p.reshape(B * C, self.n_patches, self.patch_len)
        embedded = self.embed(patches) + self.pos                 # (B*C, N, D)
        encoded = self.trunk(embedded)                            # (B*C, N, D)

        # Route each (B, c) slice to its subsystem expert.
        # Group channel indices by subsystem so we can batch each expert call.
        ch_subs = self.channel_to_subsystem_id.tolist()           # list of length C
        recon_patches = torch.empty_like(patches)
        # encoded is shaped (B*C, N, D); rows for channel c are at positions
        # [c, C+c, 2C+c, ...] (i.e., stride C starting at c).
        for sub_id in set(ch_subs):
            ch_idx = [c for c, s in enumerate(ch_subs) if s == sub_id]
            # Gather all rows belonging to this subsystem
            row_indices = []
            for b in range(B):
                base = b * C
                for c in ch_idx:
                    row_indices.append(base + c)
            row_indices_t = torch.tensor(row_indices, device=encoded.device, dtype=torch.long)
            sub_rows = encoded.index_select(0, row_indices_t)     # (N_rows, N, D)
            sub_name = self._subsystem_id_to_name[sub_id]
            expert_out = self.experts[sub_name](sub_rows)         # (N_rows, N, P)
            recon_patches.index_copy_(0, row_indices_t, expert_out)

        recon = recon_patches.reshape(B, C, T_trunc).permute(0, 2, 1).contiguous()
        if recon.shape[1] < T:
            pad = torch.zeros(B, T - recon.shape[1], C, device=recon.device)
            recon = torch.cat([recon, pad], dim=1)
        return recon


class SubsystemMoEDetector:
    """AnomalyDetector wrapping SubsystemMoEModule."""

    def __init__(
        self,
        window_length: int = 256,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        expert_hidden: int | None = None,
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
            expert_hidden=expert_hidden,
        )
        self._n_channels: int | None = int(n_channels) if n_channels is not None else None
        self.module: SubsystemMoEModule | None = None

    @property
    def name(self) -> str:
        return "subsystem_moe"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            requires_pretraining=True,
            supports_streaming=False,
            supports_multivariate=True,
            native_context_len=self._cfg["window_length"],
            output_kind=OutputKind.PER_STEP,
            supports_peft=False,
            supports_export_onnx=False,
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        recon = self.module(x)
        x_norm = (x - self.module.norm_mean) / self.module.norm_std
        return ((recon - x_norm) ** 2).mean()

    def compute_score_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        self.module.eval()
        with torch.no_grad():
            recon = self.module(x)
            x_norm = (x - self.module.norm_mean) / self.module.norm_std
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
            self.module = SubsystemMoEModule(n_channels=n_channels, **self._cfg)
        self.module.set_routing(dataset.channels)
        stats = dataset.stats()
        with torch.no_grad():
            self.module.norm_mean.copy_(torch.from_numpy(stats.means).float())
            self.module.norm_std.copy_(torch.from_numpy(stats.stds).float().clamp_min(1e-6))
        # Routing diagnostics
        from collections import Counter

        sub_counts = Counter(ch.subsystem.value for ch in dataset.channels)
        return FitResult(
            detector_name=self.name,
            mode=mode,
            n_steps=0,
            metadata={
                "n_parameters": sum(p.numel() for p in self.module.parameters()),
                "subsystem_routing": dict(sub_counts),
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
        raise NotImplementedError

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
    def load(cls, path: Path) -> SubsystemMoEDetector:
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(n_channels=data["n_channels"], **data["config"])
        det._n_channels = data["n_channels"]
        det.module = SubsystemMoEModule(n_channels=det._n_channels, **det._cfg)
        det.module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("subsystem_moe")
def _create(**kwargs: Any) -> SubsystemMoEDetector:
    return SubsystemMoEDetector(**kwargs)
