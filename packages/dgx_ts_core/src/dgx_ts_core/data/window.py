from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .schema import Channel


@dataclass(frozen=True, slots=True)
class TelemetryWindow:
    """A contiguous slice of multivariate telemetry.

    The only data type that crosses the dataset ↔ model boundary. All
    numeric arrays are numpy so this module stays free of any deep-learning
    framework dependency.

    Shapes:
        tensor:     (T, C) float32
        timestamps: (T,)   int64 (epoch ms)
        mask:       (T, C) bool   — True means the value is missing/invalid
        labels:     (T,)   bool   — True means anomaly at that step (eval only)
    """

    tensor: npt.NDArray[np.float32]
    timestamps: npt.NDArray[np.int64]
    channels: tuple[Channel, ...]
    mask: npt.NDArray[np.bool_] | None = None
    labels: npt.NDArray[np.bool_] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    # Phase 6 multi-task: extra per-step label arrays keyed by task name.
    # Backward-compatible: existing code that builds windows without this
    # field still works; the multi-task wrapper consumes when present.
    aux_labels: dict[str, npt.NDArray[Any]] | None = None

    def __post_init__(self) -> None:
        if self.tensor.ndim != 2:
            raise ValueError(f"tensor must be 2-D (T, C), got shape {self.tensor.shape}")
        t, c = self.tensor.shape
        if self.timestamps.shape != (t,):
            raise ValueError(f"timestamps shape {self.timestamps.shape} != ({t},)")
        if len(self.channels) != c:
            raise ValueError(
                f"channel count {len(self.channels)} != tensor channels {c}"
            )
        if self.mask is not None and self.mask.shape != (t, c):
            raise ValueError(f"mask shape {self.mask.shape} != ({t}, {c})")
        if self.labels is not None and self.labels.shape != (t,):
            raise ValueError(f"labels shape {self.labels.shape} != ({t},)")

    @property
    def length(self) -> int:
        return self.tensor.shape[0]

    @property
    def num_channels(self) -> int:
        return self.tensor.shape[1]
