from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class Metric(Protocol):
    """A scalar metric for anomaly detection evaluation.

    Implementations may use just the binary predictions, or also the raw
    scores (e.g., for AUC-style metrics).
    """

    @property
    def name(self) -> str: ...

    @property
    def higher_is_better(self) -> bool: ...

    def __call__(
        self,
        y_true: npt.NDArray[np.bool_],
        y_pred: npt.NDArray[np.bool_],
        scores: npt.NDArray[np.float32] | None = None,
    ) -> float: ...
