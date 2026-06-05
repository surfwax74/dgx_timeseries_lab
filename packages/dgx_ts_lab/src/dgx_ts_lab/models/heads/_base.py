"""TaskHead base class.

A head turns a per-step encoder output ``(B, T, D)`` into a task-specific
output (logits, regression value, etc.) and knows how to compute its own
loss + metrics against labels living in ``batch["aux_labels"][<task_key>]``.

The multi-task wrapper iterates over all attached heads, sums weighted
losses, and surfaces per-task metrics in the FitResult.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """Base class for a multi-task head.

    Subclasses must override ``forward``, ``compute_loss``, and
    ``compute_metrics``. They typically also set class-level ``name`` and
    ``label_key`` so the wrapper knows which entry in
    ``batch["aux_labels"]`` to consume.
    """

    #: short string identifying this head (used in metrics + logs)
    name: str = "task_head"
    #: key under ``batch["aux_labels"]`` where this head's targets live
    label_key: str = ""
    #: weight when summed into the multi-task loss
    default_loss_weight: float = 1.0

    def __init__(self, loss_weight: float | None = None) -> None:
        super().__init__()
        self.loss_weight = float(
            loss_weight if loss_weight is not None else self.default_loss_weight
        )

    # ── subclass contract ────────────────────────────────────────────────

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """encoded: (B, T, D) per-step embeddings from the shared encoder.

        Returns task-specific output (logits, regression value, etc.).
        """
        raise NotImplementedError

    def compute_loss(
        self, encoded: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Scalar loss for this head. Reads targets from
        ``batch["aux_labels"][self.label_key]``."""
        raise NotImplementedError

    def compute_metrics(
        self, encoded: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """Per-task metric dict (e.g., {"fault_acc": 0.83})."""
        raise NotImplementedError

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_targets(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        aux = batch.get("aux_labels")
        if aux is None or self.label_key not in aux:
            raise KeyError(
                f"{self.name}: batch is missing aux_labels[{self.label_key!r}]. "
                "Confirm the dataset emits multi-task labels (see "
                "datasets/synthetic/layered/labels.py)."
            )
        return aux[self.label_key]
