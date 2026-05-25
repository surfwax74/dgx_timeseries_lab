"""PatchTST + MAE — channel-independent patch transformer with masked
reconstruction objective for time-series anomaly detection.

Re-implemented from:
    Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
    with Transformers" (ICLR 2023) — the PatchTST backbone.

MAE-style training adapted for AD:
    1. Patch the input window per channel into N patches of length P.
    2. Linearly embed each patch to d_model.
    3. Mask a random fraction of patches with a learned mask token. The
       fraction is sampled per-batch from ``U(0, mask_ratio)`` so the
       encoder also sees low-/no-mask inputs and learns clean
       reconstruction; otherwise eval (apply_mask=False) hits a train/eval
       distribution shift and reconstruction error explodes on real inputs.
    4. Transformer encoder predicts the original patches.
    5. Loss: MSE on masked patches; falls back to all-patches when the
       sampled ratio collapses to zero so the step still produces gradient.
    6. Inference: no masking — reconstruction error per step is the score.

Channel-independent: each channel goes through the encoder in the batch
dimension. Simpler than channel-mixing and competitive on most TS tasks.
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


class PatchTSTMAEModule(nn.Module):
    """nn.Module for PatchTST+MAE.

    Holds per-channel normalization buffers so Fabric moves them to the
    right device automatically.
    """

    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        mask_ratio: float = 0.4,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        # Upper bound: per-batch we sample the actual mask ratio from U(0, mask_ratio).
        self.mask_ratio = float(mask_ratio)
        self.n_patches = self.seq_len // self.patch_len
        if self.n_patches < 1:
            raise ValueError(
                f"seq_len={seq_len} too small for patch_len={patch_len}"
            )

        self.embed = nn.Linear(self.patch_len, self.d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, self.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.recon_head = nn.Linear(self.d_model, self.patch_len)

        # Normalization (per channel) — registered as buffers so Fabric moves them.
        self.register_buffer("norm_mean", torch.zeros(n_channels))
        self.register_buffer("norm_std", torch.ones(n_channels))

    # ── helpers ──────────────────────────────────────────────────────────

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B*C, N, P) — channel-independent patching."""
        B, T, C = x.shape
        T_trunc = self.n_patches * self.patch_len
        x = x[:, :T_trunc, :]
        x = x.permute(0, 2, 1).contiguous()                          # (B, C, T)
        return x.reshape(B * C, self.n_patches, self.patch_len)      # (B*C, N, P)

    def unpatchify(self, patches: torch.Tensor, B: int, C: int) -> torch.Tensor:
        """(B*C, N, P) → (B, T, C)."""
        x = patches.reshape(B, C, self.n_patches * self.patch_len)
        return x.permute(0, 2, 1).contiguous()

    def _random_mask(
        self, BC: int, ratio: float, device: torch.device
    ) -> torch.Tensor:
        """Returns (BC, N) boolean — True where masked."""
        n_keep = max(1, int(self.n_patches * (1.0 - ratio)))
        noise = torch.rand(BC, self.n_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        mask = torch.ones(BC, self.n_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_keep, False)
        return mask

    # ── forward ──────────────────────────────────────────────────────────

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.norm_mean) / self.norm_std

    def forward(
        self, x: torch.Tensor, apply_mask: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x: (B, T, C). Returns (reconstructed_norm, mask | None).

        ``apply_mask=True`` during training; False during scoring.
        """
        B, _T, C = x.shape
        x_norm = self.normalize(x)
        patches = self.patchify(x_norm)                              # (B*C, N, P)
        embedded = self.embed(patches) + self.pos                    # (B*C, N, D)

        mask: torch.Tensor | None = None
        if apply_mask:
            ratio = float(torch.rand((), device=x.device).item()) * self.mask_ratio
            mask = self._random_mask(B * C, ratio, x.device)         # (B*C, N)
            mask_exp = mask.unsqueeze(-1).expand_as(embedded)
            embedded = torch.where(
                mask_exp, self.mask_token.expand_as(embedded), embedded
            )

        encoded = self.encoder(embedded)                             # (B*C, N, D)
        recon_patches = self.recon_head(encoded)                     # (B*C, N, P)
        recon_norm = self.unpatchify(recon_patches, B, C)            # (B, T, C)
        return recon_norm, mask


class PatchTSTMAEDetector:
    """AnomalyDetector wrapping PatchTSTMAEModule."""

    def __init__(
        self,
        window_length: int = 256,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        mask_ratio: float = 0.4,
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
            mask_ratio=float(mask_ratio),
        )
        self._n_channels: int | None = (
            int(n_channels) if n_channels is not None else None
        )
        self.module: PatchTSTMAEModule | None = None  # set by fit() or load()

    @property
    def name(self) -> str:
        return "patchtst_mae"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            requires_pretraining=True,
            supports_streaming=False,
            supports_multivariate=True,
            native_context_len=self._cfg["window_length"],
            output_kind=OutputKind.PER_STEP,
            supports_peft=False,
            supports_export_threshold_baked=True,
            supports_export_onnx=True,
        )

    # ── neural-detector contract ────────────────────────────────────────

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        recon_norm, mask = self.module(x, apply_mask=True)
        x_norm = self.module.normalize(x)
        # Loss on masked patches only.
        x_patches = self.module.patchify(x_norm)
        recon_patches = self.module.patchify(recon_norm)
        sq = (recon_patches - x_patches) ** 2
        # When the per-batch ratio collapses to 0, mask is all-False — fall
        # back to all-patches loss so the step still trains clean reconstruction.
        if mask is not None and bool(mask.any()):
            mask_exp = mask.unsqueeze(-1).expand_as(sq).float()
            return (sq * mask_exp).sum() / mask_exp.sum()
        return sq.mean()

    def compute_score_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        self.module.eval()
        with torch.no_grad():
            recon_norm, _ = self.module(x, apply_mask=False)
            x_norm = self.module.normalize(x)
            err = ((recon_norm - x_norm) ** 2).mean(dim=-1)  # (B, T)
        return err

    # ── AnomalyDetector Protocol ────────────────────────────────────────

    def fit(
        self,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: dict[str, Any],
    ) -> FitResult:
        """Builds the module + normalization buffers. Trainer.fit() drives
        the actual Fabric loop after calling this."""
        if self._n_channels is None:
            self._n_channels = len(dataset.channels)
        if self.module is None:
            self.module = PatchTSTMAEModule(
                n_channels=self._n_channels,
                seq_len=self._cfg["window_length"],
                patch_len=self._cfg["patch_len"],
                d_model=self._cfg["d_model"],
                n_heads=self._cfg["n_heads"],
                n_layers=self._cfg["n_layers"],
                d_ff=self._cfg["d_ff"],
                dropout=self._cfg["dropout"],
                mask_ratio=self._cfg["mask_ratio"],
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
                "note": "module + norm buffers initialized; Fabric loop drives parameter updates",
                "n_parameters": sum(p.numel() for p in self.module.parameters()),
            },
        )

    def score(self, window: TelemetryWindow) -> AnomalyScore:
        if self.module is None:
            raise RuntimeError("must fit before scoring")
        device = next(self.module.parameters()).device
        x = torch.from_numpy(window.tensor).float().unsqueeze(0).to(device)
        scores = self.compute_score_batch({"x": x}).squeeze(0).cpu().numpy().astype(np.float32)
        return AnomalyScore(scores=scores)

    def embed(self, window: TelemetryWindow):
        raise NotImplementedError("patchtst_mae does not expose patch embeddings yet")

    def reconstruct(self, window: TelemetryWindow):
        if self.module is None:
            raise RuntimeError("must fit before reconstructing")
        device = next(self.module.parameters()).device
        x = torch.from_numpy(window.tensor).float().unsqueeze(0).to(device)
        self.module.eval()
        with torch.no_grad():
            recon_norm, _ = self.module(x, apply_mask=False)
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
                "n_channels": self._n_channels,
                "module_state": self.module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "PatchTSTMAEDetector":
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(n_channels=data["n_channels"], **data["config"])
        det._n_channels = data["n_channels"]
        det.module = PatchTSTMAEModule(
            n_channels=det._n_channels,
            seq_len=det._cfg["window_length"],
            patch_len=det._cfg["patch_len"],
            d_model=det._cfg["d_model"],
            n_heads=det._cfg["n_heads"],
            n_layers=det._cfg["n_layers"],
            d_ff=det._cfg["d_ff"],
            dropout=det._cfg["dropout"],
            mask_ratio=det._cfg["mask_ratio"],
        )
        det.module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("patchtst_mae")
def _create(**kwargs: Any) -> PatchTSTMAEDetector:
    return PatchTSTMAEDetector(**kwargs)
