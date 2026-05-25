"""L5 — non-stationarity.

Slow drifts and regime changes that distinguish "normal but evolving"
telemetry from anomalies. A good detector tolerates these; a bad one
flags them. Including them in the synthetic dataset forces models to
learn the distinction.
"""

from __future__ import annotations

import numpy as np

from .component import Component, GenState


class LinearDrift(Component):
    """Sensor aging: linear offset accumulating over days."""

    kind = "linear_drift"

    def __init__(self, channel: str, drift_per_day: float) -> None:
        self.channel = channel
        self.drift_per_day = float(drift_per_day)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        days = state.t / 86400.0
        state.data[:, idx] += (self.drift_per_day * days).astype(np.float32)


class ExponentialAging(Component):
    """Multiplicative aging — e.g., battery capacity loss approaching an
    asymptote ``final_fraction`` with time constant ``time_constant_days``."""

    kind = "exponential_aging"

    def __init__(
        self,
        channel: str,
        final_fraction: float = 0.95,
        time_constant_days: float = 30.0,
    ) -> None:
        self.channel = channel
        self.final_fraction = float(final_fraction)
        self.tau_days = float(time_constant_days)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        days = state.t / 86400.0
        factor = self.final_fraction + (1.0 - self.final_fraction) * np.exp(
            -days / self.tau_days
        )
        state.data[:, idx] *= factor.astype(np.float32)


class SeasonalModulation(Component):
    """Slow seasonal sinusoidal modulation (annual cycle by default)."""

    kind = "seasonal_modulation"

    def __init__(
        self,
        channel: str,
        amplitude: float,
        period_days: float = 365.25,
        phase: float = 0.0,
    ) -> None:
        self.channel = channel
        self.amplitude = float(amplitude)
        self.period_days = float(period_days)
        self.phase = float(phase)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        days = state.t / 86400.0
        state.data[:, idx] += (
            self.amplitude * np.sin(2.0 * np.pi * days / self.period_days + self.phase)
        ).astype(np.float32)


class RegimeChange(Component):
    """Single step change at a given time — used for parameter
    reconfigurations, orbit-raise maneuvers, hardware switches."""

    kind = "regime_change"

    def __init__(self, channel: str, step_time_s: float, step_magnitude: float) -> None:
        self.channel = channel
        self.step_time_s = float(step_time_s)
        self.step_magnitude = float(step_magnitude)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        mask = state.t >= self.step_time_s
        state.data[mask, idx] += np.float32(self.step_magnitude)
