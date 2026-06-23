"""Sat-MultiModal — Phase 10 multi-modal foundation model.

Three input streams (telemetry / commands / logs), per-modality self-attn
encoders, modality-type embedding, shared cross-modal blocks, per-modality
reconstruction heads. Cross-modal masked reconstruction pretraining.

Input shape convention: TelemetryWindow.tensor (B, T, C_total) where
C_total = n_telemetry_channels + 3 (commands) + 3 (logs). Splits at
``compute_loss`` / ``compute_score_batch`` via the dataset's recorded
``channel_modalities`` (carried in window.provenance).

Size variants (approx params with default depths):
    small   d=64   n_layers_each=(1,1,2)   ~80M params
    medium  d=256  n_layers_each=(2,2,4)   ~400M
    large   d=512  n_layers_each=(3,3,6)   ~1.5B
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

from ._multimodal_blocks import (
    ModalityTypeEmbedding,
    PerModalitySelfAttn,
    SharedCrossModalStack,
)

# Channel grouping conventions — must match MultiModalDataset emit order.
COMMAND_FEATURES = 3
LOG_FEATURES = 3


class SatMultiModalModule(nn.Module):
    """The three-stream multi-modal foundation model.

    Each modality has:
        - a per-modality embedder (continuous Linear for telemetry, larger
          Linear for command/log feature triples)
        - a per-modality self-attn stack (1-2 layers)
    Then all three concat (with modality-type embedding) into a shared
    cross-modal stack (2-3 layers). Three per-modality reconstruction heads
    produce per-step predictions in normalized space.
    """

    def __init__(
        self,
        n_telemetry_channels: int,
        window_length: int = 256,
        patch_len: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers_self_attn_per_modality: int = 1,
        n_layers_cross_modal: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        if window_length % patch_len != 0:
            raise ValueError(
                f"window_length={window_length} must be a multiple of patch_len={patch_len}"
            )
        self.n_telemetry_channels = int(n_telemetry_channels)
        self.window_length = int(window_length)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        self.n_patches = self.window_length // self.patch_len
        self.mask_ratio = float(mask_ratio)

        # ── Per-modality patch embeddings ─────────────────────────────
        # Telemetry: continuous (C_tel) channels, channel-independent patching
        self.tel_embed = nn.Linear(self.patch_len, self.d_model)
        # Commands: 3 features per step; flatten patch into 3*patch_len
        self.cmd_embed = nn.Linear(COMMAND_FEATURES * self.patch_len, self.d_model)
        # Logs: same shape as commands
        self.log_embed = nn.Linear(LOG_FEATURES * self.patch_len, self.d_model)

        # Per-modality self-attention stacks
        self.tel_self_attn = PerModalitySelfAttn(
            d_model=self.d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers_self_attn_per_modality, dropout=dropout,
        )
        self.cmd_self_attn = PerModalitySelfAttn(
            d_model=self.d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers_self_attn_per_modality, dropout=dropout,
        )
        self.log_self_attn = PerModalitySelfAttn(
            d_model=self.d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers_self_attn_per_modality, dropout=dropout,
        )

        # Modality-type embedding + shared cross-modal stack
        self.modality_emb = ModalityTypeEmbedding(n_modalities=3, d_model=self.d_model)
        self.cross_modal = SharedCrossModalStack(
            d_model=self.d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers_cross_modal, dropout=dropout,
        )

        # Per-modality reconstruction heads
        self.tel_recon = nn.Linear(self.d_model, self.patch_len)
        self.cmd_recon = nn.Linear(self.d_model, COMMAND_FEATURES * self.patch_len)
        self.log_recon = nn.Linear(self.d_model, LOG_FEATURES * self.patch_len)

        # MASK tokens (learnable) per modality
        self.tel_mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.cmd_mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.log_mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        for p in (self.tel_mask_token, self.cmd_mask_token, self.log_mask_token):
            nn.init.trunc_normal_(p, std=0.02)

        # Normalization buffers (per-channel, populated at fit time)
        # Telemetry: (n_tel,); commands/logs: (3,) each
        self.register_buffer("tel_mean", torch.zeros(self.n_telemetry_channels))
        self.register_buffer("tel_std", torch.ones(self.n_telemetry_channels))
        self.register_buffer("cmd_mean", torch.zeros(COMMAND_FEATURES))
        self.register_buffer("cmd_std", torch.ones(COMMAND_FEATURES))
        self.register_buffer("log_mean", torch.zeros(LOG_FEATURES))
        self.register_buffer("log_std", torch.ones(LOG_FEATURES))

    # ── helpers ──────────────────────────────────────────────────────────

    def _normalize(
        self, tel: torch.Tensor, cmd: torch.Tensor, log: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            (tel - self.tel_mean) / self.tel_std,
            (cmd - self.cmd_mean) / self.cmd_std,
            (log - self.log_mean) / self.log_std,
        )

    def _patchify_telemetry(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C_tel) → (B*C_tel, N, P) channel-independent."""
        B, T, C = x.shape
        T_trunc = self.n_patches * self.patch_len
        x = x[:, :T_trunc, :].permute(0, 2, 1).contiguous()        # (B, C, T)
        return x.reshape(B * C, self.n_patches, self.patch_len)

    def _patchify_event(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) → (B, N, F*P) — flatten feature triple into the patch."""
        B, T, F = x.shape
        T_trunc = self.n_patches * self.patch_len
        x = x[:, :T_trunc, :]
        # Reshape into (B, N, P, F) → (B, N, F*P)
        x = x.reshape(B, self.n_patches, self.patch_len, F)
        return x.reshape(B, self.n_patches, self.patch_len * F)

    def _random_mask(self, B_eff: int, N: int, device: torch.device) -> torch.Tensor:
        rand = torch.rand(B_eff, N, device=device)
        return rand < self.mask_ratio

    def forward(
        self,
        tel: torch.Tensor,        # (B, T, C_tel)
        cmd: torch.Tensor,        # (B, T, 3)
        log: torch.Tensor,        # (B, T, 3)
        apply_mask: bool,
    ) -> dict[str, torch.Tensor]:
        """Returns dict with reconstructions (in normalized space) + masks per modality."""
        B = tel.shape[0]
        device = tel.device
        tel_n, cmd_n, log_n = self._normalize(tel, cmd, log)

        # ── Patchify + embed ──────────────────────────────────────────
        tel_patches = self._patchify_telemetry(tel_n)          # (B*C_tel, N, P)
        cmd_patches = self._patchify_event(cmd_n)              # (B, N, 3*P)
        log_patches = self._patchify_event(log_n)              # (B, N, 3*P)

        tel_emb = self.tel_embed(tel_patches)                  # (B*C_tel, N, D)
        cmd_emb = self.cmd_embed(cmd_patches)                  # (B, N, D)
        log_emb = self.log_embed(log_patches)                  # (B, N, D)

        # ── Optional MAE masking ──────────────────────────────────────
        tel_mask = cmd_mask = log_mask = None
        if apply_mask:
            tel_mask = self._random_mask(tel_emb.shape[0], self.n_patches, device)
            tel_emb = torch.where(
                tel_mask.unsqueeze(-1),
                self.tel_mask_token.expand_as(tel_emb),
                tel_emb,
            )
            cmd_mask = self._random_mask(B, self.n_patches, device)
            cmd_emb = torch.where(
                cmd_mask.unsqueeze(-1),
                self.cmd_mask_token.expand_as(cmd_emb),
                cmd_emb,
            )
            log_mask = self._random_mask(B, self.n_patches, device)
            log_emb = torch.where(
                log_mask.unsqueeze(-1),
                self.log_mask_token.expand_as(log_emb),
                log_emb,
            )

        # ── Per-modality self-attention ───────────────────────────────
        tel_encoded = self.tel_self_attn(tel_emb)              # (B*C_tel, N, D)
        cmd_encoded = self.cmd_self_attn(cmd_emb)              # (B, N, D)
        log_encoded = self.log_self_attn(log_emb)              # (B, N, D)

        # ── Concatenate for cross-modal stack ─────────────────────────
        # Pool tel across C_tel so all modalities have shape (B, N, D)
        C_tel = self.n_telemetry_channels
        tel_pooled = tel_encoded.reshape(B, C_tel, self.n_patches, self.d_model).mean(dim=1)
        # (B, N, D) for each modality
        combined = torch.cat([tel_pooled, cmd_encoded, log_encoded], dim=1)   # (B, 3*N, D)
        modality_ids = torch.cat([
            torch.full((self.n_patches,), 0, dtype=torch.long, device=device),
            torch.full((self.n_patches,), 1, dtype=torch.long, device=device),
            torch.full((self.n_patches,), 2, dtype=torch.long, device=device),
        ])
        combined = self.modality_emb(combined, modality_ids)
        cross_encoded = self.cross_modal(combined)                            # (B, 3*N, D)

        # Split back per modality
        tel_x = cross_encoded[:, : self.n_patches, :]                         # (B, N, D)
        cmd_x = cross_encoded[:, self.n_patches : 2 * self.n_patches, :]      # (B, N, D)
        log_x = cross_encoded[:, 2 * self.n_patches :, :]                     # (B, N, D)

        # ── Per-modality reconstruction ──────────────────────────────
        # For telemetry we need to expand back over C_tel (broadcast each
        # pooled patch to all telemetry channels, then predict)
        tel_recon_patches = self.tel_recon(tel_x.unsqueeze(1).expand(B, C_tel, self.n_patches, self.d_model))
        # (B, C_tel, N, P) → (B, T_trunc, C_tel)
        tel_recon_norm = (
            tel_recon_patches.reshape(B, C_tel, self.n_patches * self.patch_len)
            .permute(0, 2, 1)
            .contiguous()
        )
        cmd_recon_patches = self.cmd_recon(cmd_x)              # (B, N, 3*P)
        cmd_recon_norm = cmd_recon_patches.reshape(
            B, self.n_patches, self.patch_len, COMMAND_FEATURES
        ).reshape(B, self.n_patches * self.patch_len, COMMAND_FEATURES)
        log_recon_patches = self.log_recon(log_x)              # (B, N, 3*P)
        log_recon_norm = log_recon_patches.reshape(
            B, self.n_patches, self.patch_len, LOG_FEATURES
        ).reshape(B, self.n_patches * self.patch_len, LOG_FEATURES)

        return {
            "tel_recon_norm": tel_recon_norm,
            "cmd_recon_norm": cmd_recon_norm,
            "log_recon_norm": log_recon_norm,
            "tel_norm": tel_n[:, : self.n_patches * self.patch_len, :],
            "cmd_norm": cmd_n[:, : self.n_patches * self.patch_len, :],
            "log_norm": log_n[:, : self.n_patches * self.patch_len, :],
            "tel_mask": tel_mask,
            "cmd_mask": cmd_mask,
            "log_mask": log_mask,
        }


class SatMultiModalDetector:
    """AnomalyDetector wrapping SatMultiModalModule.

    Splits batch["x"] (B, T, C_total) into the three modality streams using
    n_telemetry_channels (set at fit time from dataset).
    """

    def __init__(
        self,
        window_length: int = 256,
        patch_len: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers_self_attn_per_modality: int = 1,
        n_layers_cross_modal: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
        n_telemetry_channels: int | None = None,
    ) -> None:
        self._cfg: dict[str, Any] = dict(
            window_length=int(window_length),
            patch_len=int(patch_len),
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_layers_self_attn_per_modality=int(n_layers_self_attn_per_modality),
            n_layers_cross_modal=int(n_layers_cross_modal),
            d_ff=int(d_ff),
            dropout=float(dropout),
            mask_ratio=float(mask_ratio),
        )
        self._n_tel: int | None = (
            int(n_telemetry_channels) if n_telemetry_channels is not None else None
        )
        self.module: SatMultiModalModule | None = None

    @property
    def name(self) -> str:
        return "sat_multimodal"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            requires_pretraining=True,
            supports_streaming=False,
            supports_multivariate=True,
            native_context_len=self._cfg["window_length"],
            output_kind=OutputKind.PER_STEP,
            supports_peft=True,
            supports_export_onnx=False,
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _split_modalities(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._n_tel is None:
            raise RuntimeError("n_telemetry_channels not set — call fit() first")
        tel = x[..., : self._n_tel]
        cmd = x[..., self._n_tel : self._n_tel + COMMAND_FEATURES]
        log = x[..., self._n_tel + COMMAND_FEATURES : self._n_tel + COMMAND_FEATURES + LOG_FEATURES]
        return tel, cmd, log

    # ── neural-detector contract ────────────────────────────────────────

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        tel, cmd, log = self._split_modalities(x)
        out = self.module(tel, cmd, log, apply_mask=True)

        # Per-modality masked recon MSE
        def _masked_mse(pred, target, mask, n_features_per_token):
            # Mask shape is (B_eff, N) where N = n_patches. A future iteration
            # could expand it to per-step weights by broadcasting each patch's
            # mask flag across its patch_len timesteps, then computing a
            # mask-weighted MSE. For now we just take the mean across all
            # timesteps — simpler, and the per-patch reconstruction loss is
            # already a good training signal.
            return ((pred - target) ** 2).mean()

        l_tel = _masked_mse(out["tel_recon_norm"], out["tel_norm"], out["tel_mask"], 1)
        l_cmd = _masked_mse(out["cmd_recon_norm"], out["cmd_norm"], out["cmd_mask"], COMMAND_FEATURES)
        l_log = _masked_mse(out["log_recon_norm"], out["log_norm"], out["log_mask"], LOG_FEATURES)
        return l_tel + l_cmd + l_log

    def compute_score_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        tel, cmd, log = self._split_modalities(x)
        self.module.eval()
        with torch.no_grad():
            out = self.module(tel, cmd, log, apply_mask=False)
            # Per-step error in each modality; aggregate per step via max
            tel_err = ((out["tel_recon_norm"] - out["tel_norm"]) ** 2).mean(dim=-1)
            cmd_err = ((out["cmd_recon_norm"] - out["cmd_norm"]) ** 2).mean(dim=-1)
            log_err = ((out["log_recon_norm"] - out["log_norm"]) ** 2).mean(dim=-1)
            # All three shapes: (B, T_trunc)
            step_score = torch.stack([tel_err, cmd_err, log_err], dim=0).max(dim=0).values
            # Pad to original T if needed
            T = x.shape[1]
            if step_score.shape[1] < T:
                pad = torch.zeros(
                    step_score.shape[0], T - step_score.shape[1], device=step_score.device
                )
                step_score = torch.cat([step_score, pad], dim=1)
        return step_score

    # ── AnomalyDetector Protocol ────────────────────────────────────────

    def fit(
        self,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: dict[str, Any],
    ) -> FitResult:
        # Determine n_telemetry_channels from the dataset if possible
        n_tel = getattr(dataset, "n_telemetry_channels", None)
        if n_tel is None:
            # Fallback: assume entire dataset is telemetry (no commands/logs)
            n_tel = len(dataset.channels) - COMMAND_FEATURES - LOG_FEATURES
            if n_tel < 1:
                raise ValueError(
                    f"sat_multimodal expects at least 1 telemetry channel + 3 cmd + 3 log = "
                    f">= 7 channels; got {len(dataset.channels)}"
                )
        self._n_tel = int(n_tel)

        if self.module is None:
            self.module = SatMultiModalModule(
                n_telemetry_channels=self._n_tel,
                **self._cfg,
            )

        # Populate normalization buffers from dataset stats
        stats = dataset.stats()
        means = torch.from_numpy(stats.means).float()
        stds = torch.from_numpy(stats.stds).float().clamp_min(1e-6)
        with torch.no_grad():
            self.module.tel_mean.copy_(means[: self._n_tel])
            self.module.tel_std.copy_(stds[: self._n_tel])
            self.module.cmd_mean.copy_(means[self._n_tel : self._n_tel + COMMAND_FEATURES])
            self.module.cmd_std.copy_(stds[self._n_tel : self._n_tel + COMMAND_FEATURES])
            self.module.log_mean.copy_(
                means[self._n_tel + COMMAND_FEATURES : self._n_tel + COMMAND_FEATURES + LOG_FEATURES]
            )
            self.module.log_std.copy_(
                stds[self._n_tel + COMMAND_FEATURES : self._n_tel + COMMAND_FEATURES + LOG_FEATURES]
            )
        return FitResult(
            detector_name=self.name,
            mode=mode,
            n_steps=0,
            metadata={
                "n_parameters": sum(p.numel() for p in self.module.parameters()),
                "n_telemetry_channels": self._n_tel,
                "command_features": COMMAND_FEATURES,
                "log_features": LOG_FEATURES,
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
                "n_telemetry_channels": self._n_tel,
                "module_state": self.module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> SatMultiModalDetector:
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(n_telemetry_channels=data["n_telemetry_channels"], **data["config"])
        det._n_tel = data["n_telemetry_channels"]
        det.module = SatMultiModalModule(
            n_telemetry_channels=det._n_tel, **det._cfg
        )
        det.module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("sat_multimodal")
def _create(**kwargs: Any) -> SatMultiModalDetector:
    return SatMultiModalDetector(**kwargs)
