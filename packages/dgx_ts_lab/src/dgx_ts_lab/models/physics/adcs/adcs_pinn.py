"""ADCSPinn — trainable PINN for attitude evolution using one of 3 integrators.

For Phase 9 we treat ADCS as a forward PINN: given an initial state +
control torque profile, train a network to predict attitude (and body rates)
at arbitrary future time. The "ground truth" trajectory is produced by the
chosen analytical integrator (Euler / RK4 / SO(3)).

This is the simplest demonstration of "PINN compressing an expensive solver
into a fast neural surrogate." For real ADCS control work, replace this with
a closed-loop RL setup using the same integrators as the simulator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from .._pinn_base_torch import PINNBackbone
from .dynamics_euler import step_euler
from .dynamics_rk4 import step_rk4
from .dynamics_so3 import step_so3
from .state import AttitudeState, SpacecraftInertia

_INTEGRATORS: dict[str, Callable] = {
    "euler": step_euler,
    "rk4": step_rk4,
    "so3": step_so3,
}


@dataclass
class ADCSPinnConfig:
    integrator: str = "rk4"             # "euler" | "rk4" | "so3"
    n_freqs: int = 16
    hidden: int = 64
    n_layers: int = 4
    horizon_s: float = 600.0
    n_collocation: int = 256


class ADCSPinn(nn.Module):
    """Forward PINN that predicts AttitudeState as a function of time.

    Output: quaternion (4) + body rates (3) + (optionally) reaction-wheel speeds.
    For Phase 9 simplicity we treat RW speeds as fixed inputs (passed via inertia config).
    """

    def __init__(
        self,
        config: ADCSPinnConfig,
        inertia: SpacecraftInertia | None = None,
    ) -> None:
        super().__init__()
        if config.integrator not in _INTEGRATORS:
            raise ValueError(
                f"unknown integrator {config.integrator!r}; "
                f"choose from {list(_INTEGRATORS)}"
            )
        self.config = config
        self.inertia = inertia or SpacecraftInertia.default_3axis()
        # Output: 7 floats (quat + body rates). RW speeds excluded for Phase 9.
        self.backbone = PINNBackbone(
            output_dim=7,
            n_freqs=config.n_freqs,
            hidden=config.hidden,
            n_layers=config.n_layers,
            min_period_s=1.0,
            max_period_s=max(10.0, config.horizon_s),
        )

    def forward(self, t: torch.Tensor) -> AttitudeState:
        """t: (B,) or (B, 1) — returns AttitudeState (B, 7) split into pieces."""
        raw = self.backbone(t)                      # (B, 7)
        q = raw[..., :4]
        # Normalize quaternion output
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        omega = raw[..., 4:7]
        rw = torch.zeros(raw.shape[0], self.inertia.rw_axes.shape[0], device=raw.device)
        return AttitudeState(q, omega, rw)

    @property
    def integrator_step(self) -> Callable:
        return _INTEGRATORS[self.config.integrator]

    def simulate_ground_truth(
        self,
        initial_state: AttitudeState,
        control_torque: torch.Tensor,
        times: torch.Tensor,
    ) -> AttitudeState:
        """Run the analytical integrator forward through ``times`` to produce GT.

        ``times`` should be 1-D evenly spaced; control_torque is broadcast across.
        Returns concatenated AttitudeState with batch dim = len(times).
        """
        step = self.integrator_step
        states = [initial_state]
        dt_list = (times[1:] - times[:-1]).tolist()
        cur = initial_state
        for dt in dt_list:
            cur = step(cur, control_torque, self.inertia, float(dt))
            states.append(cur)
        # Stack along batch dim
        q = torch.stack([s.quaternion.squeeze(0) for s in states], dim=0)
        w = torch.stack([s.body_rates.squeeze(0) for s in states], dim=0)
        rw = torch.stack([s.rw_speeds.squeeze(0) for s in states], dim=0)
        return AttitudeState(q, w, rw)
