"""First-order Euler integrator for ADCS attitude dynamics.

Simplest of the three. Cheapest per step. Susceptible to quaternion norm drift
+ energy non-conservation on long horizons — fine for short windows / demos.
"""

from __future__ import annotations

import torch

from .state import AttitudeState, SpacecraftInertia


def _quat_derivative(q: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """q_dot = 0.5 * Omega(omega) ⊗ q  (scalar-last convention).

    q:     (B, 4) [x, y, z, w]
    omega: (B, 3) body rates rad/s
    """
    # Quaternion product convention: scalar-last
    # qdot = 0.5 * q ⊗ [omega; 0]
    wx, wy, wz = omega.unbind(dim=-1)
    qx, qy, qz, qw = q.unbind(dim=-1)
    dx = 0.5 * (qw * wx + qy * wz - qz * wy)
    dy = 0.5 * (qw * wy + qz * wx - qx * wz)
    dz = 0.5 * (qw * wz + qx * wy - qy * wx)
    dw = -0.5 * (qx * wx + qy * wy + qz * wz)
    return torch.stack([dx, dy, dz, dw], dim=-1)


def _body_rate_derivative(
    omega: torch.Tensor,
    rw_speeds: torch.Tensor,
    inertia: SpacecraftInertia,
    control_torque: torch.Tensor,
) -> torch.Tensor:
    """Euler's equation: I·ω̇ = τ - ω × (I·ω + Σ h_rw)

    omega:           (B, 3)
    rw_speeds:       (B, N)
    control_torque:  (B, 3) externally-applied torque (Nm)
    """
    # Body-frame angular momentum including reaction wheels
    I_omega = omega @ inertia.inertia_body.T                      # (B, 3)
    h_rw = (rw_speeds.unsqueeze(-1) * inertia.rw_axes.unsqueeze(0)
            * inertia.rw_inertia.view(1, -1, 1)).sum(dim=1)        # (B, 3)
    h_total = I_omega + h_rw
    # Cross product ω × h
    cross = torch.cross(omega, h_total, dim=-1)
    rhs = control_torque - cross
    # Solve I·ω̇ = rhs → ω̇ = I^-1 · rhs
    I_inv = torch.linalg.inv(inertia.inertia_body)
    return rhs @ I_inv.T


def step_euler(
    state: AttitudeState,
    control_torque: torch.Tensor,
    inertia: SpacecraftInertia,
    dt: float,
) -> AttitudeState:
    """One Euler step. Returns new AttitudeState (quaternion re-normalized)."""
    q_dot = _quat_derivative(state.quaternion, state.body_rates)
    omega_dot = _body_rate_derivative(
        state.body_rates, state.rw_speeds, inertia, control_torque
    )
    new_q = state.quaternion + dt * q_dot
    new_omega = state.body_rates + dt * omega_dot
    # Reaction wheels coast (no friction model) in this baseline integrator
    return AttitudeState(new_q, new_omega, state.rw_speeds).normalize_quaternion()
