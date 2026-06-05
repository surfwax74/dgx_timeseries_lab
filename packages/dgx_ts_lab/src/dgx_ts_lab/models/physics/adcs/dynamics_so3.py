"""Lie-group SO(3) integrator — manifold-preserving attitude propagation.

Body rates are integrated via Euler (or trapezoid); attitude is updated by
multiplying the current rotation by exp(ω · dt) — staying on the SO(3)
manifold without normalization drift.

Uses quaternion exponential to compute exp(ω · dt) since we store attitude
as a quaternion. The result is exact rotation composition, not a linear
approximation, so this integrator is preferred for long-horizon simulations.
"""

from __future__ import annotations

import torch

from .dynamics_euler import _body_rate_derivative
from .state import AttitudeState, SpacecraftInertia


def _exp_quat_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    """exp map: rotation vector (B, 3) → unit quaternion (B, 4) scalar-last.

    Returns the quaternion representing rotation by angle = ||rotvec|| around
    the unit vector rotvec / ||rotvec||.
    """
    angle = rotvec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    half = 0.5 * angle
    axis = rotvec / angle
    sin_half = torch.sin(half)
    xyz = axis * sin_half
    w = torch.cos(half)
    return torch.cat([xyz, w], dim=-1)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2 (scalar-last convention)."""
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def step_so3(
    state: AttitudeState,
    control_torque: torch.Tensor,
    inertia: SpacecraftInertia,
    dt: float,
) -> AttitudeState:
    """One Lie-group step.

    Strategy: trapezoidal integration of body rates, then exponential map
    application to the quaternion. Quaternion stays unit-norm without
    explicit renormalization.
    """
    # 1. Half-step rate prediction (Euler half-step on omega)
    omega_dot_t = _body_rate_derivative(
        state.body_rates, state.rw_speeds, inertia, control_torque
    )
    omega_half = state.body_rates + 0.5 * dt * omega_dot_t

    # 2. Attitude update via exp_so3
    rotvec = omega_half * dt
    dq = _exp_quat_rotvec(rotvec)
    new_q = _quat_multiply(state.quaternion, dq)

    # 3. Full-step rate update using mid-point omega for stability
    omega_dot_mid = _body_rate_derivative(
        omega_half, state.rw_speeds, inertia, control_torque
    )
    new_omega = state.body_rates + dt * omega_dot_mid

    return AttitudeState(new_q, new_omega, state.rw_speeds)
