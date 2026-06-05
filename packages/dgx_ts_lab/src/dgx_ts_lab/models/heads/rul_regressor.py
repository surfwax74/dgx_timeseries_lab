"""RULRegressorHead — per-step regression of Remaining Useful Life.

Targets are log1p-transformed seconds-to-next-fault (clipped at a large
ceiling for "no fault in window"). Loss is masked MSE: steps where RUL is
infinite (no upcoming fault) are excluded from the loss to avoid forcing
the model to learn an arbitrary ceiling value.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgx_ts_core.registry import HEAD_REGISTRY

from ._base import TaskHead

# Matches the label generator's ceiling marker — see datasets/synthetic/layered/labels.py
NO_FAULT_CEILING_SECONDS = 1e9


class RULRegressorHead(TaskHead):
    name = "rul_regressor"
    label_key = "rul"
    default_loss_weight = 0.5    # smaller weight — different scale than classification

    def __init__(
        self,
        d_model: int = 128,
        hidden: int | None = None,
        loss_weight: float | None = None,
    ) -> None:
        super().__init__(loss_weight=loss_weight)
        h = hidden or max(64, d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        # (B, T, D) → (B, T) log1p(seconds)
        return self.net(encoded).squeeze(-1)

    @staticmethod
    def _to_log_targets(rul_seconds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (log_targets, valid_mask). valid_mask=False where rul is at the ceiling."""
        valid = rul_seconds < (NO_FAULT_CEILING_SECONDS / 10.0)
        log_targets = torch.log1p(rul_seconds.clamp_min(0.0))
        return log_targets, valid

    def compute_loss(
        self, encoded: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pred = self.forward(encoded)
        targets = self._get_targets(batch).float()
        log_t, valid = self._to_log_targets(targets)
        if not valid.any():
            # Whole batch has no upcoming faults — return zero to avoid NaN
            return pred.sum() * 0.0
        return F.mse_loss(pred[valid], log_t[valid])

    @torch.no_grad()
    def compute_metrics(
        self, encoded: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        pred = self.forward(encoded)
        targets = self._get_targets(batch).float()
        log_t, valid = self._to_log_targets(targets)
        if not valid.any():
            return {f"{self.name}.mae_log_s": float("nan"), f"{self.name}.coverage": 0.0}
        mae = float((pred[valid] - log_t[valid]).abs().mean().item())
        return {
            f"{self.name}.mae_log_s": mae,
            f"{self.name}.coverage": float(valid.float().mean().item()),
        }


@HEAD_REGISTRY.register("rul_regressor")
def _create(**kwargs: Any) -> RULRegressorHead:
    return RULRegressorHead(**kwargs)
