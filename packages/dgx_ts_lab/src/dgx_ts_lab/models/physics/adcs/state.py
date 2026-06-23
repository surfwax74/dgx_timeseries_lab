"""AttitudeState dataclass — common state representation across the three integrators.

Quaternion convention: scalar-last [x, y, z, w] (matches scipy.spatial.transform).
Body rates in spacecraft body frame, rad/s.
Reaction-wheel angular momentum stored as scalar speeds along their fixed axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttitudeState:
    """Batched spacecraft attitude state.

    Shapes (B is batch):
        quaternion:    (B, 4) — scalar-last [x, y, z, w], unit norm
        body_rates:    (B, 3) — angular velocity in body frame, rad/s
        rw_speeds:     (B, N) — N reaction-wheel speeds, rad/s
    """

    quaternion: torch.Tensor
    body_rates: torch.Tensor
    rw_speeds: torch.Tensor

    def normalize_quaternion(self) -> AttitudeState:
        """Renormalize the quaternion to unit length (drifts under Euler/RK4)."""
        q = self.quaternion / self.quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return AttitudeState(q, self.body_rates, self.rw_speeds)

    def detach_(self) -> AttitudeState:
        """In-place detach all tensors (useful between integrator steps)."""
        return AttitudeState(
            self.quaternion.detach(),
            self.body_rates.detach(),
            self.rw_speeds.detach(),
        )


@dataclass
class SpacecraftInertia:
    """Inertia properties needed for ADCS dynamics."""

    # (3, 3) body-frame inertia tensor (kg·m²)
    inertia_body: torch.Tensor
    # (N, 3) unit axes per reaction wheel in the body frame
    rw_axes: torch.Tensor
    # (N,) per-wheel inertia (kg·m²)
    rw_inertia: torch.Tensor

    @classmethod
    def default_3axis(cls, device: torch.device | None = None) -> SpacecraftInertia:
        """A reasonable small-sat default: 1 kg·m² each axis, 3 wheels on body axes."""
        dev = device or torch.device("cpu")
        # `I` matches the standard moment-of-inertia notation from rigid-body
        # mechanics (Wertz, Hughes, et al.); reads naturally to ADCS engineers.
        I = torch.diag(torch.tensor([1.0, 1.0, 1.0], device=dev))   # noqa: E741
        axes = torch.eye(3, device=dev)
        wheel_I = torch.tensor([0.01, 0.01, 0.01], device=dev)
        return cls(I, axes, wheel_I)
