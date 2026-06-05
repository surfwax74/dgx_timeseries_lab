"""Phase 9 ADCS dynamics — three integrator variants behind a common API.

    state.py            AttitudeState dataclass (quaternion + body rates + reaction wheels)
    dynamics_euler.py   First-order Euler step (cheapest, can drift)
    dynamics_rk4.py     Runge-Kutta 4th order (better accuracy, ~2x cost)
    dynamics_so3.py     Lie-group SO(3) integrator (manifold-preserving)
    adcs_pinn.py        PINN wrapper using one of the above

Pickable via config:
    integrator: "euler" | "rk4" | "so3"
"""

from .adcs_pinn import ADCSPinn
from .dynamics_euler import step_euler
from .dynamics_rk4 import step_rk4
from .dynamics_so3 import step_so3
from .state import AttitudeState, SpacecraftInertia

__all__ = [
    "ADCSPinn",
    "AttitudeState",
    "SpacecraftInertia",
    "step_euler",
    "step_rk4",
    "step_so3",
]
