from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..data.dataset import TelemetryDataset
from ..data.window import TelemetryWindow
from .capabilities import Capabilities, FitMode
from .scores import AnomalyScore


@dataclass(frozen=True, slots=True)
class FitResult:
    """Returned by AnomalyDetector.fit() and Trainer.fit()."""

    detector_name: str
    mode: FitMode
    final_loss: float | None = None
    n_steps: int = 0
    artifacts: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AnomalyDetector(Protocol):
    """The detector contract.

    Every model — from-scratch transformer, foundation-model adapter,
    physics residual wrapper, classical baseline — implements this same
    interface. Hot-swap is achieved by training and eval code branching on
    .capabilities rather than on isinstance() of the concrete class.
    """

    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> Capabilities: ...

    def fit(
        self,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: dict[str, Any],
    ) -> FitResult: ...

    def score(self, window: TelemetryWindow) -> AnomalyScore: ...

    def embed(self, window: TelemetryWindow) -> Any:
        """Optional. Implementations without latent embeddings may raise
        NotImplementedError."""
        ...

    def reconstruct(self, window: TelemetryWindow) -> Any:
        """Optional. Only reconstruction-based detectors implement this."""
        ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "AnomalyDetector": ...
