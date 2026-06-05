"""ModePredictorHead — per-step categorical prediction of the spacecraft
mode at horizon ``H`` steps ahead.

Useful for downlink scheduling + power budgeting (predict eclipse entry
before it happens). Targets come from the dataset's mode trace, shifted
by H steps.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgx_ts_core.registry import HEAD_REGISTRY

from ._base import TaskHead


# Default mode vocabulary size — matches MODE_VOCAB in
# datasets/synthetic/layered/modes.py
DEFAULT_NUM_MODES = 6


class ModePredictorHead(TaskHead):
    name = "mode_predictor"
    label_key = "next_mode"
    default_loss_weight = 0.5

    def __init__(
        self,
        d_model: int = 128,
        num_modes: int = DEFAULT_NUM_MODES,
        hidden: int | None = None,
        loss_weight: float | None = None,
    ) -> None:
        super().__init__(loss_weight=loss_weight)
        h = hidden or max(64, d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, num_modes),
        )
        self.num_modes = int(num_modes)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.net(encoded)  # (B, T, M)

    def compute_loss(
        self, encoded: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        logits = self.forward(encoded)
        targets = self._get_targets(batch).long()
        B, T, M = logits.shape
        # Some steps near the end of a window have no horizon target (-1
        # sentinel). Mask them out.
        valid = targets >= 0
        if not valid.any():
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits.reshape(B * T, M)[valid.reshape(-1)],
            targets.reshape(B * T)[valid.reshape(-1)],
        )

    @torch.no_grad()
    def compute_metrics(
        self, encoded: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        logits = self.forward(encoded)
        targets = self._get_targets(batch).long()
        pred = logits.argmax(dim=-1)
        valid = targets >= 0
        if not valid.any():
            return {f"{self.name}.acc": float("nan")}
        acc = float((pred[valid] == targets[valid]).float().mean().item())
        return {f"{self.name}.acc": acc}


@HEAD_REGISTRY.register("mode_predictor")
def _create(**kwargs: Any) -> ModePredictorHead:
    return ModePredictorHead(**kwargs)
