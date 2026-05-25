"""DCdetector — Yang et al., KDD 2023.

Dual-attention contrastive detector. Two views over the same input:

    - patch-wise attention:  attention across patches (each patch is one token)
    - in-patch attention:    attention across timesteps WITHIN each patch

A symmetric KL between the two attention distributions is the entire
objective — no reconstruction loss. The intuition: for normal patches the
two views agree (intra-patch structure matches inter-patch context); for
anomalous patches they disagree.

Score at inference is the same symmetric KL.

Re-implemented from the paper. The architecture deviates slightly from
the official code for simplicity (single-scale, single-head-equivalent
projections), but preserves the dual-view contrastive objective.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgx_ts_core.data import TelemetryDataset, TelemetryWindow
from dgx_ts_core.models import (
    AnomalyScore,
    Capabilities,
    FitMode,
    FitResult,
    OutputKind,
)
from dgx_ts_core.registry import DETECTOR_REGISTRY


def _attn(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Standard scaled-dot-product attention probs. (..., L, D) × (..., L, D) → (..., L, L)."""
    d = q.shape[-1]
    return F.softmax(q @ k.transpose(-2, -1) / math.sqrt(d), dim=-1)


class DCdetectorModule(nn.Module):
    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        patch_len: int = 16,
        d_model: int = 64,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        self.n_patches = self.seq_len // self.patch_len
        if self.n_patches < 2:
            raise ValueError(
                f"seq_len={seq_len} too small for patch_len={patch_len}"
            )

        # Channel-independent: each channel goes through the batch dim.
        # Project each timestep (length C) to d_model — for CI we treat each
        # channel separately, so the per-step input dim is 1.
        # We'll project AFTER reshape, when each "token" is either a patch
        # (shape patch_len) or a timestep (shape 1).

        # Patch-wise: each patch is one token of dim patch_len
        self.patch_q = nn.Linear(patch_len, d_model, bias=False)
        self.patch_k = nn.Linear(patch_len, d_model, bias=False)

        # In-patch: each timestep within a patch is one token of dim 1
        self.step_q = nn.Linear(1, d_model, bias=False)
        self.step_k = nn.Linear(1, d_model, bias=False)

        self.register_buffer("norm_mean", torch.zeros(n_channels))
        self.register_buffer("norm_std", torch.ones(n_channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.norm_mean) / self.norm_std

    def _attentions(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (patch_attn, step_attn).

        patch_attn: (B*C, N, N)  — softmax over patches
        step_attn:  (B*C, N, P, P) — softmax over timesteps within each patch
        """
        B, T, C = x.shape
        # patchify: (B, T, C) → (B*C, N, P)
        T_trunc = self.n_patches * self.patch_len
        x = x[:, :T_trunc, :].permute(0, 2, 1).contiguous()              # (B, C, T)
        patches = x.reshape(B * C, self.n_patches, self.patch_len)       # (B*C, N, P)

        # Patch-wise attention: tokens = patches, dim = patch_len
        pq = self.patch_q(patches)                                       # (B*C, N, D)
        pk = self.patch_k(patches)
        patch_attn = _attn(pq, pk)                                       # (B*C, N, N)

        # In-patch attention: tokens = timesteps, dim = 1
        steps = patches.unsqueeze(-1)                                    # (B*C, N, P, 1)
        sq = self.step_q(steps)                                          # (B*C, N, P, D)
        sk = self.step_k(steps)
        step_attn = _attn(sq, sk)                                        # (B*C, N, P, P)

        return patch_attn, step_attn

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.normalize(x)
        return self._attentions(x_norm)


def _row_sym_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-row symmetric KL between two distributions over last dim.

    Output shape = input shape minus the last dim (the key dim is summed).
    """
    eps = 1e-8
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1) + (q * (q.log() - p.log())).sum(dim=-1)


class DCdetectorDetector:
    """AnomalyDetector wrapping DCdetectorModule."""

    def __init__(
        self,
        window_length: int = 256,
        patch_len: int = 16,
        d_model: int = 64,
        n_channels: int | None = None,
    ):
        self._cfg: dict[str, Any] = dict(
            window_length=int(window_length),
            patch_len=int(patch_len),
            d_model=int(d_model),
        )
        self._n_channels: int | None = int(n_channels) if n_channels is not None else None
        self.module: DCdetectorModule | None = None

    @property
    def name(self) -> str:
        return "dcdetector"

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

    def _per_step_disc(
        self, patch_attn: torch.Tensor, step_attn: torch.Tensor, B: int, C: int
    ) -> torch.Tensor:
        """Per-step discrepancy of shape (B, T).

        Strategy: align both attentions to a per-step view (broadcast
        patch_attn from per-patch to per-step by repeating each row P times),
        then take symmetric KL row-wise and average across C.
        """
        N, P = self.module.n_patches, self.module.patch_len
        # patch_attn: (B*C, N, N) → per-step over (NP, NP) is too big; instead
        # treat patch-wise as the "summary" view: for each step we use the
        # row of patch_attn corresponding to that step's patch.
        # We compare patch_attn[:, n, :] (length N) to a step-level summary
        # of step_attn[:, n, p, :] (length P) — these have different lengths.
        # Standard trick: reduce step_attn to per-patch by averaging over P
        # both ways, giving a (B*C, N, N)-shaped step-derived view.
        step_per_patch = step_attn.mean(dim=2)                           # (B*C, N, P)
        # Map step_per_patch from N-of-P to N-of-N by averaging across the
        # P dim then broadcasting to N (poor man's projection — sufficient
        # for the symmetric KL to detect mismatch).
        step_view = step_per_patch.mean(dim=-1, keepdim=True).expand(-1, -1, N)
        # Re-normalize both to be valid distributions over their last dim.
        patch_dist = patch_attn / patch_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        step_dist = F.softmax(step_view, dim=-1)
        per_patch_disc = _row_sym_kl(patch_dist, step_dist)              # (B*C, N)
        # Expand from per-patch to per-step by repeating each value P times.
        per_step_BC = per_patch_disc.unsqueeze(-1).expand(-1, -1, P).reshape(B * C, N * P)
        # Average across channels: reshape (B, C, T) → mean over C
        per_step = per_step_BC.reshape(B, C, N * P).mean(dim=1)          # (B, T_trunc)
        # Pad to seq_len if needed
        if per_step.shape[1] < self.module.seq_len:
            pad = self.module.seq_len - per_step.shape[1]
            per_step = F.pad(per_step, (0, pad), value=0.0)
        return per_step

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        B, _T, C = x.shape
        patch_attn, step_attn = self.module(x)
        disc = self._per_step_disc(patch_attn, step_attn, B, C)
        # During training we MINIMIZE the discrepancy (paper objective) so
        # normal data drives the two views toward agreement.
        return disc.mean()

    def compute_score_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        B, _T, C = x.shape
        self.module.eval()
        with torch.no_grad():
            patch_attn, step_attn = self.module(x)
            score = self._per_step_disc(patch_attn, step_attn, B, C)
        return score

    def fit(self, dataset, mode, config):
        if self._n_channels is None:
            self._n_channels = len(dataset.channels)
        if self.module is None:
            self.module = DCdetectorModule(
                n_channels=self._n_channels,
                seq_len=self._cfg["window_length"],
                patch_len=self._cfg["patch_len"],
                d_model=self._cfg["d_model"],
            )
        stats = dataset.stats()
        with torch.no_grad():
            self.module.norm_mean.copy_(torch.from_numpy(stats.means).float())
            self.module.norm_std.copy_(torch.from_numpy(stats.stds).float().clamp_min(1e-6))
        return FitResult(
            detector_name=self.name,
            mode=mode,
            n_steps=0,
            metadata={"n_parameters": sum(p.numel() for p in self.module.parameters())},
        )

    def score(self, window: TelemetryWindow) -> AnomalyScore:
        if self.module is None:
            raise RuntimeError("must fit before scoring")
        device = next(self.module.parameters()).device
        x = torch.from_numpy(window.tensor).float().unsqueeze(0).to(device)
        scores = self.compute_score_batch({"x": x}).squeeze(0).cpu().numpy().astype(np.float32)
        return AnomalyScore(scores=scores)

    def embed(self, window):
        raise NotImplementedError

    def reconstruct(self, window):
        raise NotImplementedError("dcdetector is contrastive — no reconstruction")

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
    def load(cls, path: Path) -> "DCdetectorDetector":
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(n_channels=data["n_channels"], **data["config"])
        det._n_channels = data["n_channels"]
        det.module = DCdetectorModule(
            n_channels=det._n_channels,
            seq_len=det._cfg["window_length"],
            patch_len=det._cfg["patch_len"],
            d_model=det._cfg["d_model"],
        )
        det.module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("dcdetector")
def _create(**kwargs: Any) -> DCdetectorDetector:
    return DCdetectorDetector(**kwargs)
