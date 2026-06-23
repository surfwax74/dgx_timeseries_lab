from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SplitStrategy(StrEnum):
    TEMPORAL = "temporal"
    """Chronological cut — first train_frac of time → train, etc.
    Default for time-series; preserves temporal causality."""

    TEMPORAL_ANOMALY_PRESERVING = "temporal_anomaly_preserving"
    """Chronological cut, but ensures each split contains a fair share of
    labeled anomalies. Used when anomalies are rare and clustered."""

    RANDOM = "random"
    """Random per-window assignment. Generally inappropriate for time-series
    AD (leaks future into past) — exposed only for ablations."""


@dataclass(frozen=True, slots=True)
class SplitScheme:
    strategy: SplitStrategy
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if not 0.999 < total < 1.001:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")
