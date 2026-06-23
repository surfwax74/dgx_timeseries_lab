"""FaultClassifierHead — per-step categorical classification over fault types.

Class 0 is reserved for "no fault active". Remaining classes are taken from
the layered-synth fault catalog: point, dropout, stuck_at, drift,
oscillation, correlation_break, mode_confusion.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgx_ts_core.registry import HEAD_REGISTRY

from ._base import TaskHead

# Class index 0 == "no fault active". Keep in sync with
# datasets/synthetic/layered/labels.py::FAULT_CLASS_INDEX.
DEFAULT_NUM_CLASSES = 8


class FaultClassifierHead(TaskHead):
    name = "fault_classifier"
    label_key = "fault_type"
    default_loss_weight = 1.0

    def __init__(
        self,
        d_model: int = 128,
        num_classes: int = DEFAULT_NUM_CLASSES,
        hidden: int | None = None,
        loss_weight: float | None = None,
        # Weight the "no fault" class down to combat heavy imbalance.
        no_fault_weight: float = 0.1,
    ) -> None:
        super().__init__(loss_weight=loss_weight)
        h = hidden or max(64, d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, num_classes),
        )
        # CrossEntropyLoss class weights
        weights = torch.ones(num_classes)
        weights[0] = float(no_fault_weight)
        self.register_buffer("class_weights", weights)
        self.num_classes = int(num_classes)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        # (B, T, D) → (B, T, K)
        return self.net(encoded)

    def compute_loss(
        self, encoded: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        logits = self.forward(encoded)               # (B, T, K)
        targets = self._get_targets(batch).long()    # (B, T)
        # Cross-entropy over flattened (B*T, K) vs (B*T,)
        B, T, K = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, K),
            targets.reshape(B * T),
            weight=self.class_weights,
        )

    @torch.no_grad()
    def compute_metrics(
        self, encoded: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        logits = self.forward(encoded)
        targets = self._get_targets(batch).long()
        pred = logits.argmax(dim=-1)                 # (B, T)
        # Accuracy + non-zero-class macro recall (the interesting half)
        acc = float((pred == targets).float().mean().item())
        any_fault = targets > 0
        if any_fault.any():
            recall = float((pred[any_fault] == targets[any_fault]).float().mean().item())
        else:
            recall = float("nan")
        return {f"{self.name}.acc": acc, f"{self.name}.fault_recall": recall}


@HEAD_REGISTRY.register("fault_classifier")
def _create(**kwargs: Any) -> FaultClassifierHead:
    return FaultClassifierHead(**kwargs)
