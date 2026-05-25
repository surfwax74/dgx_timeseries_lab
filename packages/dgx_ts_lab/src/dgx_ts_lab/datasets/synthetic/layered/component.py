"""Foundations: GenState (the evolving generation context) and Component
(the base class every L1–L6 building block derives from).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GenState:
    """Mutable generation context passed through the L1–L6 component stack.

    Components mutate ``data``, ``mode``, ``labels``, and ``fault_log`` in
    place. The orchestrator owns construction and final teardown.
    """

    t: np.ndarray                          # (T,) float32 seconds since start
    data: np.ndarray                       # (T, C) float32 — the evolving telemetry
    mode: np.ndarray                       # (T,) int32 — discrete mode id; -1 = unset
    labels: np.ndarray                     # (T,) bool — any-fault label (OR over faults)
    fault_log: list[dict[str, Any]] = field(default_factory=list)
    channel_index: dict[str, int] = field(default_factory=dict)
    sample_rate_hz: float = 1.0
    mode_vocab: dict[str, int] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.t.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    def channel_idx(self, name: str) -> int:
        if name not in self.channel_index:
            raise KeyError(
                f"channel '{name}' not in dataset. "
                f"Available: {sorted(self.channel_index)}"
            )
        return self.channel_index[name]


class Component:
    """Base class for L1–L6 components.

    Subclasses override ``apply``. Components should be idempotent w.r.t.
    their declared inputs and deterministic given the same ``rng``.
    Order matters — the orchestrator applies them in the declared sequence.
    """

    #: Short kind string used in fault_log entries (faults override this).
    kind: str = "component"

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        raise NotImplementedError
