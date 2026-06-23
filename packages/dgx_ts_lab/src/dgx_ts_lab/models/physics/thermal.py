"""ThermalResidual — first-order thermal model for satellite surfaces.

For each thermal channel, predicts T(t) via Euler integration of:

    T(t+1) = T(t) + dt/tau * (T_eq(mode) - T(t))

where T_eq alternates between sun and shade equilibria based on the same
eclipse cycle the orbital model uses. Covers:

    sa_*_temp_*         (solar panel temperatures)
    bat_*_temp*         (battery cell-bank temperatures)
    pdu_*_temp, pcu_*_temp  (electronics box temperatures)

For non-temperature channels: prediction is 0.
"""

from __future__ import annotations

import re

import numpy as np
from dgx_ts_core.data import TelemetryWindow

# Per-channel-family thermal model parameters. Tuned to match the
# layered_synth `ThermalDutyCycle` defaults for our LEO EPS preset.
_THERMAL_PROFILES: dict[str, dict[str, float]] = {
    "panel": {"eq_sun": 55.0, "eq_shade": -40.0, "tau_s": 600.0, "initial": 20.0},
    "battery": {"eq_sun": 22.0, "eq_shade": 12.0, "tau_s": 1800.0, "initial": 18.0},
    "electronics": {"eq_sun": 29.0, "eq_shade": 19.0, "tau_s": 900.0, "initial": 23.0},
}


def _classify(name: str) -> str | None:
    if re.search(r"sa_[a-z]+_temp_[a-z]+", name):
        return "panel"
    if re.search(r"bat_[a-z]+_temp\d*", name):
        return "battery"
    if re.search(r"(pdu|pcu)_[a-z]+_temp", name):
        return "electronics"
    return None


class ThermalResidual:
    name = "thermal"

    def __init__(
        self,
        period_s: float = 5400.0,
        eclipse_fraction: float = 0.35,
    ) -> None:
        self.period_s = float(period_s)
        self.eclipse_fraction = float(eclipse_fraction)

    def covered_channels(self) -> set[str]:
        return {
            "sa_*_temp_*",
            "bat_*_temp*",
            "pdu_*_temp",
            "pcu_*_temp",
        }

    def _eclipse_mask(self, t_s: np.ndarray) -> np.ndarray:
        pos = (t_s % self.period_s) / self.period_s
        return np.abs(pos - 0.5) < (self.eclipse_fraction / 2.0)

    def _euler_temp(
        self, t_s: np.ndarray, profile: dict[str, float], dt_s: float
    ) -> np.ndarray:
        in_eclipse = self._eclipse_mask(t_s)
        eq = np.where(in_eclipse, profile["eq_shade"], profile["eq_sun"]).astype(np.float32)
        alpha = 1.0 - float(np.exp(-dt_s / profile["tau_s"]))
        out = np.empty_like(eq)
        out[0] = profile["initial"]
        for i in range(1, len(out)):
            out[i] = out[i - 1] + alpha * (eq[i] - out[i - 1])
        return out

    def predict(self, window: TelemetryWindow) -> np.ndarray:
        ts_ms = window.timestamps.astype(np.float64)
        t_s = ((ts_ms - ts_ms[0]) / 1000.0).astype(np.float64)
        # Estimate dt from successive timestamps (uniform assumed)
        dt_s = float(t_s[1] - t_s[0]) if len(t_s) > 1 else 1.0

        pred = np.zeros_like(window.tensor)
        # Cache per-profile predictions so we don't re-integrate for each channel of same family
        profile_preds: dict[str, np.ndarray] = {}
        for i, ch in enumerate(window.channels):
            kind = _classify(ch.name)
            if kind is None:
                continue
            if kind not in profile_preds:
                profile_preds[kind] = self._euler_temp(t_s, _THERMAL_PROFILES[kind], dt_s)
            pred[:, i] = profile_preds[kind]
        return pred
