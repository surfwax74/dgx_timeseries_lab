"""Hand-rolled PINN building blocks — no Modulus dependency.

Components:

    FourierFeatures           sin/cos projections of input time at log-spaced freqs
    PINNBackbone              Tanh-MLP regression network
    collocation_residual      autograd helper to compute time derivatives
    hybrid_pinn_loss          L = w_data * MSE(pred, target) + w_phys * MSE(residual)

The collocation_residual helper uses ``torch.autograd.grad`` to compute
``d(pred)/dt`` at requested collocation points, which subclasses then plug
into their own PDE residual function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Project scalar time → (2 * n_freqs,) sin/cos features at log-spaced freqs.

    Helps PINNs learn periodic targets (orbital effects, daily cycles).
    """

    def __init__(
        self,
        n_freqs: int = 16,
        min_period_s: float = 60.0,
        max_period_s: float = 86400.0,
    ) -> None:
        super().__init__()
        log_min = math.log(min_period_s)
        log_max = math.log(max_period_s)
        periods = torch.exp(torch.linspace(log_min, log_max, n_freqs))
        self.register_buffer("omegas", 2.0 * math.pi / periods)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1) — returns (B, 2 * n_freqs)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        wt = t * self.omegas.view(1, -1)              # (B, n_freqs)
        return torch.cat([torch.sin(wt), torch.cos(wt)], dim=-1)


class PINNBackbone(nn.Module):
    """Standard Tanh-MLP with Fourier-feature input.

    Output dim configurable per use case (n_zones for thermal, 7 for ADCS state, …).
    """

    def __init__(
        self,
        output_dim: int,
        n_freqs: int = 16,
        hidden: int = 64,
        n_layers: int = 4,
        min_period_s: float = 60.0,
        max_period_s: float = 86400.0,
    ) -> None:
        super().__init__()
        self.fourier = FourierFeatures(n_freqs, min_period_s, max_period_s)
        input_dim = 2 * n_freqs
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        features = self.fourier(t)
        return self.net(features)


def time_derivative(
    output: torch.Tensor, t: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    """Autograd-computed d(output)/dt.

    output: (B, D)
    t:      (B, 1) with requires_grad=True
    Returns (B, D).
    """
    grad_outputs = torch.ones_like(output)
    grads = torch.autograd.grad(
        outputs=output,
        inputs=t,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]                                # (B, 1) per-element d/dt summed... no.
    # The above is the correct multi-output trick: requires per-output gradient
    # When output is multi-dim, we need to compute per-dim grad separately.
    # Simpler: loop over output dim.
    if output.shape[-1] == 1:
        return grads
    # Multi-output: compute via vector-Jacobian product per col
    out_grads = []
    for i in range(output.shape[-1]):
        g_i = torch.autograd.grad(
            outputs=output[:, i],
            inputs=t,
            grad_outputs=torch.ones_like(output[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        out_grads.append(g_i)
    return torch.cat(out_grads, dim=-1)


@dataclass
class HybridLossConfig:
    """Knobs for the hybrid PINN loss."""

    w_data: float = 1.0
    w_physics: float = 1.0
    n_collocation_points: int = 256
    collocation_t_min_s: float = 0.0
    collocation_t_max_s: float = 86400.0


def hybrid_pinn_loss(
    *,
    data_loss: torch.Tensor,
    physics_residual: torch.Tensor,
    cfg: HybridLossConfig,
) -> torch.Tensor:
    """L = w_data * data_loss + w_physics * MSE(physics_residual).

    Subclasses compute the residual according to their PDE and call this.
    """
    physics_loss = (physics_residual ** 2).mean()
    return cfg.w_data * data_loss + cfg.w_physics * physics_loss


def sample_collocation_times(
    n: int, t_min_s: float, t_max_s: float, device: torch.device
) -> torch.Tensor:
    """Uniformly-sampled collocation times for physics residual eval."""
    t = torch.rand(n, 1, device=device) * (t_max_s - t_min_s) + t_min_s
    t.requires_grad_(True)
    return t
