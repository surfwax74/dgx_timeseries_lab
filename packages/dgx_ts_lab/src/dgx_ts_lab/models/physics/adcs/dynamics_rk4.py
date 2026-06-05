"""Runge-Kutta 4th-order integrator for ADCS attitude dynamics.

~2x cost of Euler per step; substantially better accuracy.
Still has quaternion-norm drift (renormalized after each step).
"""

from __future__ import annotations

import torch

from .dynamics_euler import _body_rate_derivative, _quat_derivative
from .state import AttitudeState, SpacecraftInertia


def _derivatives(
    q: torch.Tensor,
    omega: torch.Tensor,
    rw: torch.Tensor,
    inertia: SpacecraftInertia,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _quat_derivative(q, omega),
        _body_rate_derivative(omega, rw, inertia, tau),
    )


def step_rk4(
    state: AttitudeState,
    control_torque: torch.Tensor,
    inertia: SpacecraftInertia,
    dt: float,
) -> AttitudeState:
    """One RK4 step assuming the control torque is constant over [t, t+dt]."""
    q0, w0 = state.quaternion, state.body_rates
    rw = state.rw_speeds  # constant over step (no friction model)
    tau = control_torque

    # k1
    k1q, k1w = _derivatives(q0, w0, rw, inertia, tau)
    # k2
    k2q, k2w = _derivatives(q0 + 0.5 * dt * k1q, w0 + 0.5 * dt * k1w, rw, inertia, tau)
    # k3
    k3q, k3w = _derivatives(q0 + 0.5 * dt * k2q, w0 + 0.5 * dt * k2w, rw, inertia, tau)
    # k4
    k4q, k4w = _derivatives(q0 + dt * k3q, w0 + dt * k3w, rw, inertia, tau)

    new_q = q0 + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    new_w = w0 + (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
    return AttitudeState(new_q, new_w, rw).normalize_quaternion()
