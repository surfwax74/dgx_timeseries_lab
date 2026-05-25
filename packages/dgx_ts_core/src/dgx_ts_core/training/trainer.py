from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..data.dataset import TelemetryDataset
from ..evaluation.result import EvalReport
from ..models.capabilities import FitMode
from ..models.detector import AnomalyDetector, FitResult
from .config import TrainConfig


@runtime_checkable
class Trainer(Protocol):
    """Wraps Lightning Fabric (or any other engine).

    Knows how to drive a detector through a FitMode against a dataset.
    Implementations decide on distributed strategy, mixed precision,
    callbacks, and checkpointing.
    """

    def fit(
        self,
        detector: AnomalyDetector,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: TrainConfig,
    ) -> FitResult: ...

    def zero_shot(
        self,
        detector: AnomalyDetector,
        dataset: TelemetryDataset,
    ) -> EvalReport: ...
