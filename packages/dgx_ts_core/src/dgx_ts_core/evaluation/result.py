from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Standard structured output from any evaluation run.

    `metrics` is the headline scalar dict (e.g., {"f1": 0.82, "pr_auc": 0.79}).
    `per_channel` optionally breaks the same metrics down per telemetry channel.
    """

    detector_name: str
    dataset_name: str
    metrics: dict[str, float]
    threshold: float | None = None
    per_channel: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
