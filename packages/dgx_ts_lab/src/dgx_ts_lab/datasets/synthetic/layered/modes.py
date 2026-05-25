"""L2 — discrete mode signals.

Writes a discrete mode index into ``state.mode`` that downstream physics
and fault components can branch on (eclipse-driven thermal/battery
behavior, mode-confusion faults, etc.).
"""

from __future__ import annotations

import numpy as np

from .component import Component, GenState

MODE_VOCAB: dict[str, int] = {
    "sun": 0,
    "eclipse": 1,
    "payload_idle": 2,
    "payload_active": 3,
    "safe": 4,
    "transmit": 5,
}


class ModeMachine(Component):
    """Eclipse-cycle mode machine with optional payload activation overlay.

    By default writes a sun/eclipse cycle based on orbital position. If
    ``payload_active_rate_per_hour > 0``, also Poisson-samples payload
    activation windows on top of the base sun/eclipse mode (the last
    overlay wins per timestep).
    """

    kind = "mode_machine"

    def __init__(
        self,
        period_s: float = 5400.0,
        eclipse_fraction: float = 0.35,
        eclipse_phase: float = 0.0,
        payload_active_rate_per_hour: float = 0.0,
        payload_duration_s: float = 300.0,
        write_vocab: bool = True,
    ) -> None:
        self.period_s = float(period_s)
        self.eclipse_fraction = float(eclipse_fraction)
        self.eclipse_phase = float(eclipse_phase)
        self.payload_rate = float(payload_active_rate_per_hour)
        self.payload_duration_s = float(payload_duration_s)
        self.write_vocab = bool(write_vocab)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        # Base mode: sun (0) most of the orbit, eclipse (1) in the shadow window.
        pos = ((state.t + self.eclipse_phase) % self.period_s) / self.period_s
        in_eclipse = np.abs(pos - 0.5) < (self.eclipse_fraction / 2.0)
        base = np.where(in_eclipse, MODE_VOCAB["eclipse"], MODE_VOCAB["sun"]).astype(np.int32)

        # Optional payload activations sprinkled in.
        if self.payload_rate > 0:
            n = state.n_steps
            hours = n / (state.sample_rate_hz * 3600.0)
            expected = self.payload_rate * hours
            n_activations = int(rng.poisson(expected))
            dur_steps = max(1, int(self.payload_duration_s * state.sample_rate_hz))
            for _ in range(n_activations):
                start = int(rng.integers(0, max(1, n - dur_steps)))
                base[start : start + dur_steps] = MODE_VOCAB["payload_active"]

        state.mode[:] = base
        if self.write_vocab:
            state.mode_vocab.update(MODE_VOCAB)
