"""OperatorFingerprintDetector — per-operator Mahalanobis-distance AD.

Phase 8 cybersecurity detector for D12 "behavior fingerprint" anomalies.
Trains a small encoder over activity-window features; at fit time computes
per-operator embedding statistics (mean + covariance inverse) on the
training data; at score time embeds new windows and returns the
Mahalanobis distance to the CLAIMED operator's distribution.

If the embedding for window W under claim "alice" is far from alice's
training-time distribution, that's an anomaly (likely impersonation).
"""

from __future__ import annotations

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


class OperatorFingerprintModule(nn.Module):
    """Encoder MLP + per-operator Gaussian stats (mean + cov_inv buffers)."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 16,
        n_operators: int = 4,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.n_operators = int(n_operators)

        h = hidden_dim or max(32, 2 * input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, h),
            nn.GELU(),
            nn.Linear(h, self.embedding_dim),
        )

        # Per-operator statistics, populated at fit time.
        self.register_buffer(
            "operator_means", torch.zeros(self.n_operators, self.embedding_dim)
        )
        # Init cov_inv with identity matrices so untrained detectors still produce finite scores.
        self.register_buffer(
            "operator_cov_inv",
            torch.eye(self.embedding_dim).unsqueeze(0).expand(self.n_operators, -1, -1).contiguous().clone(),
        )
        # Per-operator validity flag (set to 1.0 once stats are populated)
        self.register_buffer("operator_valid", torch.zeros(self.n_operators))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) — returns (B, embedding_dim) per-window embedding via temporal mean."""
        # Mean-pool across time first, then encode
        return self.encoder(x.mean(dim=1))

    def embed_per_step(self, x: torch.Tensor) -> torch.Tensor:
        """Per-step embeddings for per-position scoring. (B, T, F) → (B, T, D)."""
        B, T, _ = x.shape
        flat = x.reshape(B * T, -1)
        embedded = self.encoder(flat)
        return embedded.reshape(B, T, self.embedding_dim)

    def mahalanobis_distance(
        self, embeddings: torch.Tensor, operator_ids: torch.Tensor
    ) -> torch.Tensor:
        """Per-step Mahalanobis distance.

        embeddings: (B, T, D)
        operator_ids: (B, T) long — which operator each step is claimed under
        Returns: (B, T) float — Mahalanobis distance to that operator's distribution
        """
        op_ids = operator_ids.clamp(0, self.n_operators - 1)
        means = self.operator_means[op_ids]              # (B, T, D)
        cov_invs = self.operator_cov_inv[op_ids]         # (B, T, D, D)
        diffs = embeddings - means                       # (B, T, D)
        # m^2 = diff @ cov_inv @ diff^T  (per row)
        m_sq = torch.einsum("btd,btde,bte->bt", diffs, cov_invs, diffs).clamp_min(0.0)
        return torch.sqrt(m_sq + 1e-8)


class OperatorFingerprintDetector:
    """AnomalyDetector wrapping the per-operator Mahalanobis distance model."""

    def __init__(
        self,
        embedding_dim: int = 16,
        n_operators: int = 4,
        hidden_dim: int | None = None,
        n_channels: int | None = None,
        # Reconstruction-loss weight during pretraining (helps train the encoder
        # before Mahalanobis stats exist).
        pretrain_contrastive_weight: float = 1.0,
    ) -> None:
        self._embedding_dim = int(embedding_dim)
        self._n_operators = int(n_operators)
        self._hidden_dim = hidden_dim
        self._n_channels = int(n_channels) if n_channels is not None else None
        self._pretrain_weight = float(pretrain_contrastive_weight)
        self.module: OperatorFingerprintModule | None = None

    @property
    def name(self) -> str:
        return "operator_fingerprint"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            requires_pretraining=True,
            supports_streaming=True,       # cheap inference once trained
            supports_multivariate=True,    # input is multi-feature activity vectors
            native_context_len=64,
            output_kind=OutputKind.PER_STEP,
            supports_peft=False,
            supports_export_onnx=True,
        )

    # ── neural-detector contract ────────────────────────────────────────

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Contrastive embedding loss: push same-operator embeddings together,
        different-operator apart. No Mahalanobis until fit-stats step."""
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]                              # (B, T, F)
        aux = batch.get("aux_labels", {}) or {}
        op_ids = aux.get("operator_id")
        if op_ids is None:
            # Fall back: reconstruction-style — just embed and minimize L2 norm spread
            emb = self.module.embed(x)              # (B, D)
            return ((emb - emb.mean(dim=0, keepdim=True)) ** 2).mean()
        # Per-window mean operator (use mode of the window)
        op_ids = op_ids.long()
        emb = self.module.embed(x)                  # (B, D)
        # Pair-wise distance matrix
        d = torch.cdist(emb, emb)                   # (B, B)
        # Take the window-wise majority operator
        ops_per_window = op_ids.mode(dim=1).values  # (B,)
        same = (ops_per_window.unsqueeze(0) == ops_per_window.unsqueeze(1)).float()
        # Triplet-style: minimize same-pair distance, maximize diff-pair distance
        margin = 1.0
        # Average distance for same-operator pairs
        n_same = same.sum().clamp_min(1.0)
        n_diff = (1.0 - same).sum().clamp_min(1.0)
        loss_pos = (d * same).sum() / n_same
        loss_neg = F.relu(margin - d) * (1.0 - same)
        return self._pretrain_weight * (loss_pos + loss_neg.sum() / n_diff)

    def compute_score_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]                              # (B, T, F)
        aux = batch.get("aux_labels", {}) or {}
        op_ids = aux.get("operator_id")
        if op_ids is None:
            # No claim — return zeros (caller should always provide operator_id)
            return torch.zeros(x.shape[0], x.shape[1], device=x.device)
        self.module.eval()
        with torch.no_grad():
            embeddings = self.module.embed_per_step(x)
            return self.module.mahalanobis_distance(embeddings, op_ids.long())

    # ── AnomalyDetector Protocol ────────────────────────────────────────

    def fit(
        self,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: dict[str, Any],
    ) -> FitResult:
        n_channels = len(dataset.channels)
        if self._n_channels is None:
            self._n_channels = n_channels
        n_operators = int(getattr(dataset, "n_operators", self._n_operators))
        if n_operators > self._n_operators:
            self._n_operators = n_operators
        if self.module is None:
            self.module = OperatorFingerprintModule(
                input_dim=self._n_channels,
                embedding_dim=self._embedding_dim,
                n_operators=self._n_operators,
                hidden_dim=self._hidden_dim,
            )
        # NOTE: per-operator stats are populated after the Fabric loop trains
        # the encoder. The trainer calls compute_operator_stats() at the end
        # of training (see fabric_loop integration in W5).
        return FitResult(
            detector_name=self.name,
            mode=mode,
            n_steps=0,
            metadata={
                "n_parameters": sum(p.numel() for p in self.module.parameters()),
                "n_operators": self._n_operators,
                "embedding_dim": self._embedding_dim,
            },
        )

    @torch.no_grad()
    def compute_operator_stats(
        self, dataset: TelemetryDataset, window_length: int, stride: int
    ) -> dict[int, int]:
        """Walk the training data, compute per-operator embedding mean + cov_inv.

        Called AFTER the encoder has been trained (typically end of fit).
        Returns ``{operator_id: n_samples_observed}`` for diagnostics.
        """
        assert self.module is not None
        device = next(self.module.parameters()).device
        # Bucket embeddings by operator
        buckets: dict[int, list[torch.Tensor]] = {
            op_id: [] for op_id in range(self._n_operators)
        }
        for w in dataset.windows(length=window_length, stride=stride):
            if w.aux_labels is None or "operator_id" not in w.aux_labels:
                continue
            op_window = int(np.bincount(w.aux_labels["operator_id"]).argmax())
            x = torch.from_numpy(w.tensor).float().unsqueeze(0).to(device)
            emb = self.module.embed(x).squeeze(0)  # (D,)
            if 0 <= op_window < self._n_operators:
                buckets[op_window].append(emb)

        counts: dict[int, int] = {}
        for op_id, embs in buckets.items():
            counts[op_id] = len(embs)
            if len(embs) < 2:
                continue   # leave default identity cov_inv
            stacked = torch.stack(embs, dim=0)                  # (N, D)
            mean = stacked.mean(dim=0)
            centered = stacked - mean
            cov = (centered.T @ centered) / max(1, len(embs) - 1)
            cov = cov + 1e-3 * torch.eye(self._embedding_dim, device=cov.device)
            cov_inv = torch.linalg.inv(cov)
            with torch.no_grad():
                self.module.operator_means[op_id].copy_(mean)
                self.module.operator_cov_inv[op_id].copy_(cov_inv)
                self.module.operator_valid[op_id] = 1.0
        return counts

    def score(self, window: TelemetryWindow) -> AnomalyScore:
        if self.module is None:
            raise RuntimeError("must fit before scoring")
        device = next(self.module.parameters()).device
        x = torch.from_numpy(window.tensor).float().unsqueeze(0).to(device)
        aux = {}
        if window.aux_labels is not None and "operator_id" in window.aux_labels:
            aux["operator_id"] = torch.from_numpy(
                window.aux_labels["operator_id"].copy()
            ).long().unsqueeze(0).to(device)
        scores = (
            self.compute_score_batch({"x": x, "aux_labels": aux})
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return AnomalyScore(scores=scores)

    def embed(self, window):
        raise NotImplementedError

    def reconstruct(self, window):
        raise NotImplementedError("operator_fingerprint scores via Mahalanobis, not reconstruction")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.module is None:
            raise RuntimeError("nothing to save — detector not fit")
        torch.save(
            {
                "embedding_dim": self._embedding_dim,
                "n_operators": self._n_operators,
                "hidden_dim": self._hidden_dim,
                "n_channels": self._n_channels,
                "pretrain_contrastive_weight": self._pretrain_weight,
                "module_state": self.module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> OperatorFingerprintDetector:
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(
            embedding_dim=data["embedding_dim"],
            n_operators=data["n_operators"],
            hidden_dim=data["hidden_dim"],
            n_channels=data["n_channels"],
            pretrain_contrastive_weight=data["pretrain_contrastive_weight"],
        )
        det.module = OperatorFingerprintModule(
            input_dim=det._n_channels or 6,
            embedding_dim=det._embedding_dim,
            n_operators=det._n_operators,
            hidden_dim=det._hidden_dim,
        )
        det.module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("operator_fingerprint")
def _create(**kwargs: Any) -> OperatorFingerprintDetector:
    return OperatorFingerprintDetector(**kwargs)
