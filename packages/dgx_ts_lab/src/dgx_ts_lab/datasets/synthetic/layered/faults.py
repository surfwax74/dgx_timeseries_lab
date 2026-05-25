"""L6 — fault injection.

Faults are separate from noise: they write to ``state.labels`` AND log a
structured entry to ``state.fault_log`` so eval can break results down
per fault category, severity, and dwell time.
"""

from __future__ import annotations

import numpy as np

from .component import Component, GenState
from .modes import MODE_VOCAB


def _n_events_poisson(
    rate: float, period_factor: float, rng: np.random.Generator
) -> int:
    """Sample event count from Poisson(rate * period_factor)."""
    return int(rng.poisson(max(rate * period_factor, 0.0)))


def _hours(state: GenState) -> float:
    return state.n_steps / (state.sample_rate_hz * 3600.0)


def _days(state: GenState) -> float:
    return state.n_steps / (state.sample_rate_hz * 86400.0)


class PointFault(Component):
    """Random impulse spikes at Poisson-distributed instants."""

    kind = "point_fault"

    def __init__(
        self,
        channel: str,
        rate_per_hour: float = 0.5,
        magnitude: float = 5.0,
        sign: str = "random",  # "random" | "positive" | "negative"
    ) -> None:
        self.channel = channel
        self.rate = float(rate_per_hour)
        self.magnitude = float(magnitude)
        self.sign = sign

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        n_faults = _n_events_poisson(self.rate, _hours(state), rng)
        if n_faults == 0 or state.n_steps == 0:
            return
        positions = rng.integers(0, state.n_steps, size=n_faults)
        for pos in positions:
            if self.sign == "positive":
                s = 1.0
            elif self.sign == "negative":
                s = -1.0
            else:
                s = float(rng.choice([-1.0, 1.0]))
            state.data[pos, idx] += np.float32(s * self.magnitude)
            state.labels[pos] = True
            state.fault_log.append(
                {
                    "type": self.kind,
                    "channel": self.channel,
                    "start": int(pos),
                    "end": int(pos + 1),
                    "magnitude": s * self.magnitude,
                }
            )


class DropoutFault(Component):
    """Telemetry dropouts — channel held at ``fill_value`` for a brief window."""

    kind = "dropout_fault"

    def __init__(
        self,
        channel: str,
        rate_per_hour: float = 0.1,
        min_duration_s: float = 1.0,
        max_duration_s: float = 10.0,
        fill_value: float = 0.0,
    ) -> None:
        self.channel = channel
        self.rate = float(rate_per_hour)
        self.min_dur = float(min_duration_s)
        self.max_dur = float(max_duration_s)
        self.fill = float(fill_value)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        n_faults = _n_events_poisson(self.rate, _hours(state), rng)
        if n_faults == 0:
            return
        min_steps = max(1, int(self.min_dur * state.sample_rate_hz))
        max_steps = max(min_steps, int(self.max_dur * state.sample_rate_hz))
        for _ in range(n_faults):
            duration = int(rng.integers(min_steps, max_steps + 1))
            start = int(rng.integers(0, max(1, state.n_steps - duration)))
            end = min(start + duration, state.n_steps)
            state.data[start:end, idx] = np.float32(self.fill)
            state.labels[start:end] = True
            state.fault_log.append(
                {
                    "type": self.kind,
                    "channel": self.channel,
                    "start": start,
                    "end": end,
                    "fill_value": self.fill,
                }
            )


class StuckAtFault(Component):
    """Sensor frozen — value held constant at its value at fault onset."""

    kind = "stuck_at_fault"

    def __init__(
        self,
        channel: str,
        rate_per_day: float = 0.5,
        min_duration_s: float = 60.0,
        max_duration_s: float = 600.0,
    ) -> None:
        self.channel = channel
        self.rate = float(rate_per_day)
        self.min_dur = float(min_duration_s)
        self.max_dur = float(max_duration_s)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        n_faults = _n_events_poisson(self.rate, _days(state), rng)
        if n_faults == 0:
            return
        min_steps = max(1, int(self.min_dur * state.sample_rate_hz))
        max_steps = max(min_steps, int(self.max_dur * state.sample_rate_hz))
        for _ in range(n_faults):
            duration = int(rng.integers(min_steps, max_steps + 1))
            start = int(rng.integers(0, max(1, state.n_steps - duration)))
            end = min(start + duration, state.n_steps)
            stuck_val = float(state.data[start, idx])
            state.data[start:end, idx] = np.float32(stuck_val)
            state.labels[start:end] = True
            state.fault_log.append(
                {
                    "type": self.kind,
                    "channel": self.channel,
                    "start": start,
                    "end": end,
                    "stuck_value": stuck_val,
                }
            )


class DriftFault(Component):
    """Calibration drift onset: bias ramps up over a window, then persists."""

    kind = "drift_fault"

    def __init__(
        self,
        channel: str,
        rate_per_day: float = 0.2,
        ramp_duration_s: float = 1800.0,
        final_offset: float = 1.0,
        persist: bool = True,
    ) -> None:
        self.channel = channel
        self.rate = float(rate_per_day)
        self.ramp_dur = float(ramp_duration_s)
        self.final_offset = float(final_offset)
        self.persist = bool(persist)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        n_faults = _n_events_poisson(self.rate, _days(state), rng)
        if n_faults == 0:
            return
        ramp_steps = max(1, int(self.ramp_dur * state.sample_rate_hz))
        for _ in range(n_faults):
            start = int(rng.integers(0, max(1, state.n_steps - ramp_steps)))
            end = min(start + ramp_steps, state.n_steps)
            ramp = np.linspace(0.0, self.final_offset, end - start, dtype=np.float32)
            state.data[start:end, idx] += ramp
            if self.persist and end < state.n_steps:
                state.data[end:, idx] += np.float32(self.final_offset)
            state.labels[start:end] = True
            state.fault_log.append(
                {
                    "type": self.kind,
                    "channel": self.channel,
                    "start": start,
                    "end": end,
                    "final_offset": self.final_offset,
                    "persist": self.persist,
                }
            )


class OscillationFault(Component):
    """Sudden high-frequency oscillation onset — control-system instability."""

    kind = "oscillation_fault"

    def __init__(
        self,
        channel: str,
        rate_per_day: float = 0.1,
        frequency_hz: float = 0.5,
        amplitude: float = 2.0,
        duration_s: float = 60.0,
    ) -> None:
        self.channel = channel
        self.rate = float(rate_per_day)
        self.freq = float(frequency_hz)
        self.amplitude = float(amplitude)
        self.dur_s = float(duration_s)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        n_faults = _n_events_poisson(self.rate, _days(state), rng)
        if n_faults == 0:
            return
        dur_steps = max(2, int(self.dur_s * state.sample_rate_hz))
        for _ in range(n_faults):
            start = int(rng.integers(0, max(1, state.n_steps - dur_steps)))
            end = min(start + dur_steps, state.n_steps)
            local_t = np.arange(end - start, dtype=np.float32) / state.sample_rate_hz
            osc = self.amplitude * np.sin(2.0 * np.pi * self.freq * local_t)
            state.data[start:end, idx] += osc.astype(np.float32)
            state.labels[start:end] = True
            state.fault_log.append(
                {
                    "type": self.kind,
                    "channel": self.channel,
                    "start": start,
                    "end": end,
                    "frequency_hz": self.freq,
                    "amplitude": self.amplitude,
                }
            )


class CorrelationBreakFault(Component):
    """A normally-correlated channel suddenly diverges from its reference
    (additive independent noise on top of normal signal)."""

    kind = "correlation_break_fault"

    def __init__(
        self,
        channel: str,
        reference_channel: str,
        rate_per_day: float = 0.05,
        divergence_std: float = 2.0,
        duration_s: float = 300.0,
    ) -> None:
        self.channel = channel
        self.reference = reference_channel
        self.rate = float(rate_per_day)
        self.div_std = float(divergence_std)
        self.dur_s = float(duration_s)

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        # Reference is for label/metadata only — divergence is independent noise.
        _ = state.channel_idx(self.reference)
        idx = state.channel_idx(self.channel)
        n_faults = _n_events_poisson(self.rate, _days(state), rng)
        if n_faults == 0:
            return
        dur_steps = max(1, int(self.dur_s * state.sample_rate_hz))
        for _ in range(n_faults):
            start = int(rng.integers(0, max(1, state.n_steps - dur_steps)))
            end = min(start + dur_steps, state.n_steps)
            divergence = rng.normal(0.0, self.div_std, size=end - start)
            state.data[start:end, idx] += divergence.astype(np.float32)
            state.labels[start:end] = True
            state.fault_log.append(
                {
                    "type": self.kind,
                    "channel": self.channel,
                    "reference": self.reference,
                    "start": start,
                    "end": end,
                    "divergence_std": self.div_std,
                }
            )


class ModeConfusionFault(Component):
    """Inject a value consistent with the WRONG mode for the current step.

    Example: payload power-draw level while ``mode == sun`` (idle). The
    fault overwrites the channel value to ``wrong_value`` for the
    duration, label-flagging those steps.
    """

    kind = "mode_confusion_fault"

    def __init__(
        self,
        channel: str,
        rate_per_day: float = 0.1,
        wrong_value: float = 10.0,
        duration_s: float = 120.0,
        only_when_mode: str | None = None,
    ) -> None:
        self.channel = channel
        self.rate = float(rate_per_day)
        self.wrong_value = float(wrong_value)
        self.dur_s = float(duration_s)
        self.only_when_mode = only_when_mode

    def apply(self, state: GenState, rng: np.random.Generator) -> None:
        idx = state.channel_idx(self.channel)
        n_faults = _n_events_poisson(self.rate, _days(state), rng)
        if n_faults == 0:
            return
        dur_steps = max(1, int(self.dur_s * state.sample_rate_hz))
        target_mode = None
        if self.only_when_mode is not None:
            target_mode = MODE_VOCAB.get(self.only_when_mode)
            if target_mode is None:
                raise ValueError(
                    f"unknown mode '{self.only_when_mode}'; known: {sorted(MODE_VOCAB)}"
                )
        for _ in range(n_faults):
            start = int(rng.integers(0, max(1, state.n_steps - dur_steps)))
            end = min(start + dur_steps, state.n_steps)
            if target_mode is not None and not (state.mode[start:end] == target_mode).any():
                continue
            state.data[start:end, idx] = np.float32(self.wrong_value)
            state.labels[start:end] = True
            state.fault_log.append(
                {
                    "type": self.kind,
                    "channel": self.channel,
                    "start": start,
                    "end": end,
                    "wrong_value": self.wrong_value,
                    "only_when_mode": self.only_when_mode,
                }
            )
