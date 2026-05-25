from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models.capabilities import Capabilities


@dataclass
class ModelCard:
    """Descriptive artifact emitted alongside an exported detector.

    Read by downstream MLOps systems to understand intended use, training
    data provenance, declared capabilities, and the calibrated threshold.
    Serialized to YAML as part of the lift-to-MLOps handoff.
    """

    detector_name: str
    detector_version: str
    capabilities: Capabilities
    intended_subsystem: str
    training_dataset: str
    metrics: dict[str, float] = field(default_factory=dict)
    calibrated_threshold: float | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
