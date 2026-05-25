from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..data.schema import Channel


@dataclass
class FeatureSchema:
    """Channel-level schema emitted alongside an exported detector.

    Mirrors TelemetryDataset.channels + DatasetStats so downstream feature
    pipelines in mm_mlops can stay structurally in sync with training-time
    assumptions. Serialized to YAML for the MLOps handoff.
    """

    channels: tuple[Channel, ...]
    sample_rate_hz: float
    window_length: int
    normalization_means: npt.NDArray[np.float32]
    normalization_stds: npt.NDArray[np.float32]
