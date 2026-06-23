"""BatteryResidual — coulomb counting + Nernst-style voltage model.

Predicts:

    bat_*_soc       — state of charge from integrated charge/discharge current
    bat_*_voltage   — terminal voltage as a simple linear function of SoC

Driven by the same eclipse cycle as the orbital + thermal models:
charging in sun, discharging in eclipse.
"""

from __future__ import annotations

import re

import numpy as np
from dgx_ts_core.data import TelemetryWindow


def _classify(name: str) -> str | None:
    if re.fullmatch(r"bat_[a-z]+_soc", name):
        return "soc"
    if re.fullmatch(r"bat_[a-z]+_voltage", name):
        return "voltage"
    return None


class BatteryResidual:
    name = "battery"

    def __init__(
        self,
        period_s: float = 5400.0,
        eclipse_fraction: float = 0.35,
        capacity_Ah: float = 30.0,
        charge_rate_A: float = 2.5,
        discharge_rate_A: float = 4.0,
        initial_soc: float = 0.7,
        # Linear V-vs-SoC: V = V_min + (V_max-V_min)*SoC. Loose Li-ion approximation.
        v_min: float = 27.0,
        v_max: float = 29.5,
    ) -> None:
        self.period_s = float(period_s)
        self.eclipse_fraction = float(eclipse_fraction)
        self.capacity_Ah = float(capacity_Ah)
        self.charge_rate_A = float(charge_rate_A)
        self.discharge_rate_A = float(discharge_rate_A)
        self.initial_soc = float(initial_soc)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

    def covered_channels(self) -> set[str]:
        return {"bat_*_soc", "bat_*_voltage"}

    def _eclipse_mask(self, t_s: np.ndarray) -> np.ndarray:
        pos = (t_s % self.period_s) / self.period_s
        return np.abs(pos - 0.5) < (self.eclipse_fraction / 2.0)

    def _soc(self, t_s: np.ndarray, dt_s: float) -> np.ndarray:
        in_eclipse = self._eclipse_mask(t_s)
        current = np.where(in_eclipse, -self.discharge_rate_A, self.charge_rate_A).astype(np.float32)
        dt_h = dt_s / 3600.0
        d_soc = current * dt_h / self.capacity_Ah
        out = np.empty_like(d_soc)
        out[0] = self.initial_soc
        for i in range(1, len(out)):
            out[i] = float(np.clip(out[i - 1] + d_soc[i], 0.0, 1.0))
        return out

    def predict(self, window: TelemetryWindow) -> np.ndarray:
        ts_ms = window.timestamps.astype(np.float64)
        t_s = ((ts_ms - ts_ms[0]) / 1000.0).astype(np.float64)
        dt_s = float(t_s[1] - t_s[0]) if len(t_s) > 1 else 1.0

        # Compute SoC once; reuse for all SoC + voltage channels
        soc = self._soc(t_s, dt_s)

        pred = np.zeros_like(window.tensor)
        for i, ch in enumerate(window.channels):
            kind = _classify(ch.name)
            if kind == "soc":
                pred[:, i] = soc
            elif kind == "voltage":
                pred[:, i] = self.v_min + (self.v_max - self.v_min) * soc
        return pred
