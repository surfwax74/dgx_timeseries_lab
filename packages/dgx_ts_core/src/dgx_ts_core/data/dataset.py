from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from .schema import Channel, Subsystem
from .splits import SplitScheme
from .window import TelemetryWindow


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Per-channel descriptive statistics used for normalization and export."""

    means: npt.NDArray[np.float32]   # (C,)
    stds: npt.NDArray[np.float32]    # (C,)
    p99: npt.NDArray[np.float32]     # (C,)
    n_samples: int
    n_channels: int


@runtime_checkable
class TelemetryDataset(Protocol):
    """The dataset contract.

    Any source (NASA SMAP/MSL, ESA OPS-SAT, synthetic generators, real mission
    telemetry) implements this Protocol. The only requirements are that the
    source can describe its channels, yield TelemetryWindow objects, split
    itself temporally, and report per-channel statistics.
    """

    @property
    def name(self) -> str: ...

    @property
    def subsystem(self) -> Subsystem: ...

    @property
    def channels(self) -> tuple[Channel, ...]: ...

    @property
    def sample_rate_hz(self) -> float: ...

    @property
    def has_labels(self) -> bool: ...

    def windows(self, length: int, stride: int) -> Iterator[TelemetryWindow]: ...

    def split(self, scheme: SplitScheme) -> Mapping[str, TelemetryDataset]: ...

    def stats(self) -> DatasetStats: ...
