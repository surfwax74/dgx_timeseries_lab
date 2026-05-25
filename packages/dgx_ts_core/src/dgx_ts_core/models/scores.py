from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class AnomalyScore:
    """A detector's output for a single TelemetryWindow.

    For PER_STEP detectors: `scores` has shape (T,).
    For PER_WINDOW detectors: `scores` has shape (1,).
    A higher score means more anomalous.
    """

    scores: npt.NDArray[np.float32]
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.scores.ndim != 1:
            raise ValueError(f"scores must be 1-D, got shape {self.scores.shape}")

    @property
    def is_per_step(self) -> bool:
        return self.scores.size > 1

    def to_binary(self, threshold: float | None = None) -> npt.NDArray[np.bool_]:
        t = threshold if threshold is not None else self.threshold
        if t is None:
            raise ValueError("no threshold provided or stored on AnomalyScore")
        return self.scores > t
