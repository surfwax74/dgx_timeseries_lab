"""L3 — channel coupling.

Cross-channel correlations driven by simple physics: power draw heats
electronics, current spikes cause voltage sag, attitude error couples
into thermal, etc. These components read from one channel and add a
weighted (optionally lagged) contribution to another.
"""

from __future__ import annotations

import numpy as np

from .component import Component, GenState


class LinearCoupling(Component):
    """Add ``gain * source`` to ``target``, optionally with an integer lag.

    Lag is in samples (use ``lag_steps = int(lag_s * sample_rate_hz)``).
    Positive lag means target sees source after a delay.
    """

    kind = "linear_coupling"

    def __init__(
        self,
        source: str,
        target: str,
        gain: float = 1.0,
        lag_steps: int = 0,
        offset: float = 0.0,
    ) -> None:
        self.source = source
        self.target = target
        self.gain = float(gain)
        self.lag_steps = int(lag_steps)
        self.offset = float(offset)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        s_idx = state.channel_idx(self.source)
        t_idx = state.channel_idx(self.target)
        src = state.data[:, s_idx]
        contribution = self.gain * src + self.offset
        if self.lag_steps == 0:
            state.data[:, t_idx] += contribution.astype(np.float32)
        elif self.lag_steps > 0:
            state.data[self.lag_steps :, t_idx] += contribution[: -self.lag_steps].astype(np.float32)
        else:
            shift = -self.lag_steps
            state.data[: -shift, t_idx] += contribution[shift:].astype(np.float32)


class SumCoupling(Component):
    """Add the weighted sum of N source channels into a target channel.

    Cleaner than chaining N ``LinearCoupling`` calls when you need a
    rollup like "bus current = sum of all load currents" or "panel
    current = sum of cell-string currents".

    Parameters
    ----------
    sources : list[str]
        Channel names whose values are summed.
    target : str
        Channel that receives the sum.
    gains : list[float] | None
        Per-source multiplicative gain (default: 1.0 each). Must match
        the length of ``sources`` if provided.
    offset : float
        Constant added to the sum (e.g., quiescent current floor).
    """

    kind = "sum_coupling"

    def __init__(
        self,
        sources: list[str],
        target: str,
        gains: list[float] | None = None,
        offset: float = 0.0,
    ) -> None:
        self.sources = list(sources)
        self.target = target
        self.gains = list(gains) if gains is not None else [1.0] * len(self.sources)
        self.offset = float(offset)
        if len(self.gains) != len(self.sources):
            raise ValueError(
                f"gains length {len(self.gains)} != sources length {len(self.sources)}"
            )

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        t_idx = state.channel_idx(self.target)
        total = np.full(state.n_steps, np.float32(self.offset), dtype=np.float32)
        for src, gain in zip(self.sources, self.gains):
            s_idx = state.channel_idx(src)
            total += np.float32(gain) * state.data[:, s_idx]
        state.data[:, t_idx] += total


class InverseCoupling(Component):
    """``target -= gain * source`` — useful for voltage-sag-under-load patterns."""

    kind = "inverse_coupling"

    def __init__(self, source: str, target: str, gain: float = 1.0) -> None:
        self.source = source
        self.target = target
        self.gain = float(gain)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        s_idx = state.channel_idx(self.source)
        t_idx = state.channel_idx(self.target)
        state.data[:, t_idx] -= (self.gain * state.data[:, s_idx]).astype(np.float32)
