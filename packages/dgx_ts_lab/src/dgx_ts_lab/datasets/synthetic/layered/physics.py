"""L1 — physics-driven base signals.

Deterministic per-channel signals grounded in (highly idealized) orbital
mechanics. These are deliberately not high-fidelity — they exist to give
the detector something physically plausible to learn the structure of.
"""

from __future__ import annotations

import numpy as np

from .component import Component, GenState

# Mode IDs that physics components recognize. Kept in sync with modes.py.
MODE_SUN = 0
MODE_ECLIPSE = 1


class OrbitalSinusoid(Component):
    """Sinusoidal signal at the orbital period.

    Typical LEO orbit is ~5400 s. Used for any channel whose value cycles
    with orbit position (sun angle, magnetometer X/Y, etc.).
    """

    kind = "orbital_sinusoid"

    def __init__(
        self,
        channel: str,
        amplitude: float = 1.0,
        period_s: float = 5400.0,
        phase: float = 0.0,
        baseline: float = 0.0,
    ) -> None:
        self.channel = channel
        self.amplitude = float(amplitude)
        self.period_s = float(period_s)
        self.phase = float(phase)
        self.baseline = float(baseline)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        omega = 2.0 * np.pi / self.period_s
        state.data[:, idx] += (
            self.baseline + self.amplitude * np.sin(omega * state.t + self.phase)
        ).astype(np.float32)


class SolarIllumination(Component):
    """Sun visibility fraction. 1.0 in sun, 0.0 in eclipse, with smooth tanh
    transitions to mimic atmospheric refraction at horizon crossings.

    Independent of ``ModeMachine`` so it can be used without one — pure
    function of time.
    """

    kind = "solar_illumination"

    def __init__(
        self,
        channel: str,
        period_s: float = 5400.0,
        eclipse_fraction: float = 0.35,
        transition_s: float = 60.0,
        phase: float = 0.0,
    ) -> None:
        self.channel = channel
        self.period_s = float(period_s)
        self.eclipse_fraction = float(eclipse_fraction)
        self.transition_s = float(transition_s)
        self.phase = float(phase)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        # Fractional position in orbit ∈ [0, 1)
        pos = ((state.t + self.phase) % self.period_s) / self.period_s
        # Eclipse window centered at pos = 0.5; half-width = eclipse_fraction/2
        d = np.abs(pos - 0.5) - (self.eclipse_fraction / 2.0)
        # tanh transitions: very negative d → sun (≈ 0 sin output), positive → eclipse
        illum = 0.5 * (1.0 + np.tanh(d * (self.period_s / self.transition_s)))
        state.data[:, idx] += illum.astype(np.float32)


class ConstantBaseline(Component):
    """Add a constant DC offset to a channel.

    Trivial but pervasive: many telemetry channels live around a nominal
    value (28 V bus, 5 W idle load, etc.). This is the cleanest way to set
    that nominal — clearer than using ``OrbitalSinusoid`` with amplitude=0.
    """

    kind = "constant_baseline"

    def __init__(self, channel: str, value: float = 0.0) -> None:
        self.channel = channel
        self.value = float(value)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        state.data[:, idx] += np.float32(self.value)


class ThermalDutyCycle(Component):
    """First-order thermal response. Temperature relaxes toward a
    mode-dependent equilibrium with time constant tau.

    Requires ``ModeMachine`` to have already written sun/eclipse modes
    into ``state.mode``. If no mode info is present, falls back to the
    sunlit equilibrium uniformly.
    """

    kind = "thermal_duty_cycle"

    def __init__(
        self,
        channel: str,
        equilibrium_sun_C: float = 20.0,
        equilibrium_shade_C: float = -30.0,
        time_constant_s: float = 600.0,
        initial_C: float = 0.0,
    ) -> None:
        self.channel = channel
        self.equilibrium_sun = float(equilibrium_sun_C)
        self.equilibrium_shade = float(equilibrium_shade_C)
        self.tau = float(time_constant_s)
        self.initial = float(initial_C)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        dt = 1.0 / state.sample_rate_hz
        alpha = 1.0 - np.exp(-dt / self.tau)
        target = np.where(
            state.mode == MODE_ECLIPSE, self.equilibrium_shade, self.equilibrium_sun
        ).astype(np.float32)
        # Iterative Euler — vectorizing requires solving a recurrence;
        # T steps is fine for realistic dataset sizes.
        temp = np.empty(state.n_steps, dtype=np.float32)
        temp[0] = self.initial
        for i in range(1, state.n_steps):
            temp[i] = temp[i - 1] + alpha * (target[i] - temp[i - 1])
        state.data[:, idx] += temp


class BatterySoC(Component):
    """Battery state-of-charge driven by sun/eclipse mode.

    Charges at ``charge_rate_A`` in sun, discharges at ``discharge_rate_A``
    in eclipse, clipped to [0, 1]. Stored in fractional units.
    """

    kind = "battery_soc"

    def __init__(
        self,
        channel: str,
        capacity_Ah: float = 10.0,
        charge_rate_A: float = 1.5,
        discharge_rate_A: float = 2.0,
        initial_soc: float = 0.5,
    ) -> None:
        self.channel = channel
        self.capacity_Ah = float(capacity_Ah)
        self.charge_rate_A = float(charge_rate_A)
        self.discharge_rate_A = float(discharge_rate_A)
        self.initial_soc = float(initial_soc)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        dt_h = 1.0 / (state.sample_rate_hz * 3600.0)
        current = np.where(
            state.mode == MODE_ECLIPSE, -self.discharge_rate_A, self.charge_rate_A
        ).astype(np.float32)
        d_soc = current * dt_h / self.capacity_Ah
        soc = np.empty(state.n_steps, dtype=np.float32)
        soc[0] = self.initial_soc
        for i in range(1, state.n_steps):
            soc[i] = np.clip(soc[i - 1] + d_soc[i], 0.0, 1.0)
        state.data[:, idx] += soc
