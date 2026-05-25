"""OrbitalResidual — predicts sun-angle / eclipse-driven channels.

Uses the same orbital cycle model as the L1 ``OrbitalSinusoid`` /
``SolarIllumination`` components in the layered synthetic generator. For
the LEO EPS preset this covers:

    sa_*_str*_current        (cell-string currents — illumination-driven)
    sa_*_current             (total panel current — sum of strings)
    sa_*_voltage             (panel voltage — orbital ripple)
    sa_*_sada_angle          (SADA position — tracks sun)

For any other channel the prediction is 0 (residual = data).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from dgx_ts_core.data import TelemetryWindow


class OrbitalResidual:
    """Predict orbital-cycle-driven telemetry from time alone."""

    name = "orbital"

    # Default parameters for a LEO orbit. Override per-mission if needed.
    def __init__(
        self,
        period_s: float = 5400.0,
        eclipse_fraction: float = 0.35,
        # Per-channel-family amplitudes (added to a per-channel baseline).
        string_current_baseline: float = 0.84,
        string_current_amplitude: float = 0.05,
        panel_voltage_baseline: float = 28.0,
        panel_voltage_amplitude: float = 0.4,
        sada_amplitude_rad: float = 3.14159,
    ) -> None:
        self.period_s = float(period_s)
        self.eclipse_fraction = float(eclipse_fraction)
        self.str_base = float(string_current_baseline)
        self.str_amp = float(string_current_amplitude)
        self.pv_base = float(panel_voltage_baseline)
        self.pv_amp = float(panel_voltage_amplitude)
        self.sada_amp = float(sada_amplitude_rad)

    def covered_channels(self) -> set[str]:
        # Discovered per-window — return the pattern set for diagnostics.
        return {
            "sa_*_str*_current",
            "sa_*_current",
            "sa_*_voltage",
            "sa_*_sada_angle",
        }

    @staticmethod
    def _classify(name: str) -> str | None:
        if re.fullmatch(r"sa_[a-z]+_str\d+_current", name):
            return "string"
        if re.fullmatch(r"sa_[a-z]+_current", name):
            return "panel_total"
        if re.fullmatch(r"sa_[a-z]+_voltage", name):
            return "voltage"
        if re.fullmatch(r"sa_[a-z]+_sada_angle", name):
            return "sada"
        return None

    def predict(self, window: TelemetryWindow) -> np.ndarray:
        # Build time vector in seconds from timestamps (epoch ms).
        ts_ms = window.timestamps.astype(np.float64)
        t_s = (ts_ms - ts_ms[0]) / 1000.0
        omega = 2.0 * np.pi / self.period_s

        sin_t = np.sin(omega * t_s).astype(np.float32)
        # Smooth eclipse factor: 1 in sun, 0 in eclipse, smooth at edges
        pos = ((t_s % self.period_s) / self.period_s).astype(np.float32)
        d = np.abs(pos - 0.5) - (self.eclipse_fraction / 2.0)
        illum = (0.5 * (1.0 + np.tanh(d * 50.0))).astype(np.float32)

        pred = np.zeros_like(window.tensor)
        for i, ch in enumerate(window.channels):
            kind = self._classify(ch.name)
            if kind == "string":
                pred[:, i] = illum * (self.str_base + self.str_amp * sin_t)
            elif kind == "panel_total":
                # ≈ 4 strings worth (preset has 4 strings per panel)
                pred[:, i] = illum * 4.0 * self.str_base
            elif kind == "voltage":
                pred[:, i] = self.pv_base + self.pv_amp * sin_t
            elif kind == "sada":
                pred[:, i] = self.sada_amp * sin_t
            # else: leave zero
        return pred


# Register in the local physics registry (defined in __init__.py)
def _register():
    from . import _PHYSICS_REGISTRY

    _PHYSICS_REGISTRY["orbital"] = OrbitalResidual
