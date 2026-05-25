"""Anomaly Transformer — Xu et al., ICLR 2022.

Re-implemented from the paper. Each encoder layer has two attention
distributions per timestep:

    - prior:  Gaussian kernel parameterized by a learnable per-head σ
    - series: standard scaled-dot-product attention

The "association discrepancy" is the symmetric KL between them. Loss is:

    L = MSE(reconstruction, input) ± λ · AssDisc(prior, series)

Trained with a min-max objective: the +λ phase pushes the series attention
to AGREE with the prior on normal points; the −λ phase fights this for
unstable points. We use the standard practical simplification of training
only with +λ (the paper's "double-phase" optimization complicates the loop
and the +-only version still works well).

At inference:
    anomaly_score = AssDisc(prior, series) · ||x - recon||²
The product highlights points that are BOTH hard to reconstruct AND have
attention mismatched between prior and series.
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


class _AnomalyAttention(nn.Module):
    """One layer's dual attention: prior (Gaussian) + series (standard).

    Returns (output, prior_attn, series_attn) where the two attention
    tensors have shape (B, H, T, T) for KL computation.
    """

    def __init__(self, d_model: int, n_heads: int, seq_len: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.seq_len = seq_len

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Learnable per-head sigma for the Gaussian prior; softplus keeps positive.
        self.sigma_raw = nn.Parameter(torch.zeros(n_heads))

        # Pre-compute (T, T) distance matrix; broadcast at forward time.
        idx = torch.arange(seq_len, dtype=torch.float32)
        self.register_buffer("dist", (idx[:, None] - idx[None, :]).abs())  # (T, T)

    def _prior(self) -> torch.Tensor:
        """Returns (H, T, T) prior attention via per-head Gaussian over |i-j|."""
        sigma = F.softplus(self.sigma_raw) + 1e-3                       # (H,)
        sigma = sigma.view(self.n_heads, 1, 1)                          # (H, 1, 1)
        dist = self.dist.unsqueeze(0)                                   # (1, T, T)
        logp = -(dist ** 2) / (2.0 * sigma ** 2)                        # (H, T, T)
        logp = logp - torch.log(sigma * math.sqrt(2.0 * math.pi))
        return F.softmax(logp, dim=-1)                                  # (H, T, T)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        # (B, T, H, Dh) → (B, H, T, Dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, T, T)
        series_attn = F.softmax(attn_logits, dim=-1)
        out = (series_attn @ v).transpose(1, 2).reshape(B, T, self.d_model)
        out = self.out_proj(self.dropout(out))

        prior_attn = self._prior().unsqueeze(0).expand(B, -1, -1, -1)     # (B, H, T, T)
        return out, prior_attn, series_attn


class _Block(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, seq_len: int, dropout: float
    ):
        super().__init__()
        self.attn = _AnomalyAttention(d_model, n_heads, seq_len, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, prior, series = self.attn(self.norm1(x))
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x, prior, series


class AnomalyTransformerModule(nn.Module):
    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)

        self.embed = nn.Linear(n_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [
                _Block(d_model, n_heads, d_ff, seq_len, dropout)
                for _ in range(int(n_layers))
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.recon_head = nn.Linear(d_model, n_channels)

        self.register_buffer("norm_mean", torch.zeros(n_channels))
        self.register_buffer("norm_std", torch.ones(n_channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.norm_mean) / self.norm_std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        # x: (B, T, C) — returns (recon_norm, priors, series)
        x_norm = self.normalize(x)
        h = self.embed(x_norm) + self.pos                                # (B, T, D)
        priors: list[torch.Tensor] = []
        series: list[torch.Tensor] = []
        for blk in self.blocks:
            h, p, s = blk(h)
            priors.append(p)
            series.append(s)
        recon = self.recon_head(self.out_norm(h))
        return recon, priors, series


def _symmetric_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-row symmetric KL between two distributions over last dim.

    Inputs are (B, H, T, T) attention distributions; output is (B, H, T) — the
    discrepancy of each query position. We sum the symmetric KL over the key
    dim and average over heads at the call site.
    """
    eps = 1e-8
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return ((p * (p.log() - q.log())).sum(dim=-1) + (q * (q.log() - p.log())).sum(dim=-1))


class AnomalyTransformerDetector:
    """AnomalyDetector wrapping AnomalyTransformerModule."""

    def __init__(
        self,
        window_length: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        lambda_disc: float = 3.0,
        n_channels: int | None = None,
    ):
        self._cfg: dict[str, Any] = dict(
            window_length=int(window_length),
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            d_ff=int(d_ff),
            dropout=float(dropout),
        )
        self._lambda = float(lambda_disc)
        self._n_channels: int | None = int(n_channels) if n_channels is not None else None
        self.module: AnomalyTransformerModule | None = None

    @property
    def name(self) -> str:
        return "anomaly_transformer"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            requires_pretraining=True,
            supports_streaming=False,
            supports_multivariate=True,
            native_context_len=self._cfg["window_length"],
            output_kind=OutputKind.PER_STEP,
            supports_peft=False,
            supports_export_onnx=False,  # nested per-layer attn output complicates ONNX trace
        )

    def _disc_per_step(
        self, priors: list[torch.Tensor], series: list[torch.Tensor]
    ) -> torch.Tensor:
        """Mean over layers and heads of |sym-KL(prior, series)|. Returns (B, T)."""
        per_layer = []
        for p, s in zip(priors, series):
            d = _symmetric_kl(p, s).mean(dim=1)  # (B, T) — avg over heads
            per_layer.append(d)
        return torch.stack(per_layer, dim=0).mean(dim=0)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        x_norm = self.module.normalize(x)
        recon, priors, series = self.module(x)
        recon_loss = F.mse_loss(recon, x_norm)
        disc = self._disc_per_step(priors, series).mean()
        # +λ phase only (paper's simplified variant); paper's full min-max
        # alternates sign — easy upgrade later, not needed for the bake-off.
        return recon_loss + self._lambda * disc

    def compute_score_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        x_norm = self.module.normalize(x)
        self.module.eval()
        with torch.no_grad():
            recon, priors, series = self.module(x)
            recon_err = ((recon - x_norm) ** 2).mean(dim=-1)                  # (B, T)
            disc = self._disc_per_step(priors, series)                        # (B, T)
            score = recon_err * (disc + 1.0)                                  # paper's product
        return score

    def fit(
        self,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: dict[str, Any],
    ) -> FitResult:
        if self._n_channels is None:
            self._n_channels = len(dataset.channels)
        if self.module is None:
            self.module = AnomalyTransformerModule(
                n_channels=self._n_channels,
                seq_len=self._cfg["window_length"],
                d_model=self._cfg["d_model"],
                n_heads=self._cfg["n_heads"],
                n_layers=self._cfg["n_layers"],
                d_ff=self._cfg["d_ff"],
                dropout=self._cfg["dropout"],
            )
        stats = dataset.stats()
        with torch.no_grad():
            self.module.norm_mean.copy_(torch.from_numpy(stats.means).float())
            self.module.norm_std.copy_(torch.from_numpy(stats.stds).float().clamp_min(1e-6))
        return FitResult(
            detector_name=self.name,
            mode=mode,
            n_steps=0,
            metadata={
                "n_parameters": sum(p.numel() for p in self.module.parameters()),
                "lambda_disc": self._lambda,
            },
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
        if self.module is None:
            raise RuntimeError("must fit before reconstructing")
        device = next(self.module.parameters()).device
        x = torch.from_numpy(window.tensor).float().unsqueeze(0).to(device)
        self.module.eval()
        with torch.no_grad():
            recon_norm, _, _ = self.module(x)
            recon = recon_norm * self.module.norm_std + self.module.norm_mean
        return recon.squeeze(0).cpu().numpy()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.module is None:
            raise RuntimeError("nothing to save — detector not fit")
        torch.save(
            {
                "config": self._cfg,
                "lambda_disc": self._lambda,
                "n_channels": self._n_channels,
                "module_state": self.module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "AnomalyTransformerDetector":
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(
            n_channels=data["n_channels"],
            lambda_disc=data["lambda_disc"],
            **data["config"],
        )
        det._n_channels = data["n_channels"]
        det.module = AnomalyTransformerModule(
            n_channels=det._n_channels,
            seq_len=det._cfg["window_length"],
            d_model=det._cfg["d_model"],
            n_heads=det._cfg["n_heads"],
            n_layers=det._cfg["n_layers"],
            d_ff=det._cfg["d_ff"],
            dropout=det._cfg["dropout"],
        )
        det.module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("anomaly_transformer")
def _create(**kwargs: Any) -> AnomalyTransformerDetector:
    return AnomalyTransformerDetector(**kwargs)
