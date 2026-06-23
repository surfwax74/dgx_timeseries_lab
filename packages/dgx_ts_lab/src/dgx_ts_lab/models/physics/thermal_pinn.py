"""Trainable thermal PINN — hand-rolled, no Modulus dependency.

Trains a Tanh-MLP with Fourier features to approximate the multi-zone
thermal solver's output. Uses the hybrid data + collocation loss:

    L = w_data * MSE(pred, observed_truth) + w_phys * MSE(dT/dt - solver_rhs)

The collocation residual uses ``torch.autograd.grad`` to compute dT/dt
at sampled time points and compares against the analytical solver's RHS.

After training, this implements `PhysicsModel` and slots into
``PINNResidualDetector`` to subtract physics predictions from observed
telemetry before the inner detector sees the residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from dgx_ts_core.data import TelemetryWindow

from ._pinn_base_torch import (
    HybridLossConfig,
    PINNBackbone,
    hybrid_pinn_loss,
    sample_collocation_times,
)
from ._thermal_solver import ThermalBus


@dataclass
class ThermalPinnConfig:
    n_zones: int = 6
    n_freqs: int = 24
    hidden: int = 128
    n_layers: int = 5
    min_period_s: float = 60.0
    max_period_s: float = 86400.0
    horizon_s: float = 86400.0
    # Hybrid loss weights
    w_data: float = 1.0
    w_physics: float = 0.1
    n_collocation: int = 256


class ThermalPinn(nn.Module):
    """Time → (n_zones,) temperature predictor."""

    def __init__(self, config: ThermalPinnConfig, bus: ThermalBus | None = None) -> None:
        super().__init__()
        self.config = config
        self.bus = bus or ThermalBus(n_zones=config.n_zones)
        if self.bus.n_zones != config.n_zones:
            raise ValueError(
                f"bus.n_zones ({self.bus.n_zones}) != config.n_zones ({config.n_zones})"
            )
        self.backbone = PINNBackbone(
            output_dim=config.n_zones,
            n_freqs=config.n_freqs,
            hidden=config.hidden,
            n_layers=config.n_layers,
            min_period_s=config.min_period_s,
            max_period_s=config.max_period_s,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) or (B, 1). Returns (B, n_zones) predicted temperatures (K)."""
        return self.backbone(t)

    # ── Physics residual ────────────────────────────────────────────────

    def _physics_residual_at(self, t_collocation: torch.Tensor) -> torch.Tensor:
        """Compute |dT_pred/dt - dT_solver/dt| at the collocation points.

        Uses torch.autograd for dT_pred/dt and the analytical bus dynamics
        (computed in NumPy) for dT_solver/dt.
        """
        T_pred = self.forward(t_collocation)            # (B, n_zones)
        # Compute dT_pred/dt via autograd per zone
        dT_pred_dt_cols: list[torch.Tensor] = []
        for z in range(self.config.n_zones):
            g = torch.autograd.grad(
                outputs=T_pred[:, z],
                inputs=t_collocation,
                grad_outputs=torch.ones_like(T_pred[:, z]),
                create_graph=True,
                retain_graph=True,
            )[0]                                         # (B, 1)
            dT_pred_dt_cols.append(g)
        dT_pred_dt = torch.cat(dT_pred_dt_cols, dim=-1)  # (B, n_zones)

        # Compute analytical dT/dt using current T_pred values + bus dynamics
        # (this is the "physics" the network is supposed to obey)
        dT_solver_dt = self._solver_dT_dt(T_pred.detach(), t_collocation.detach())

        return dT_pred_dt - dT_solver_dt

    def _solver_dT_dt(self, T: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Analytical dT/dt evaluated at (T, t) points via the bus model.

        Computed in NumPy then wrapped back to a Tensor (no grad).
        """
        t_np = t.detach().cpu().numpy().squeeze(-1)
        T_np = T.detach().cpu().numpy()
        out_np = np.empty_like(T_np)

        # Re-use the same orbital model as simulate() for sun_vec + eclipse
        # (consistent with what generated the training data).
        orbital_period_s = 5400.0
        eclipse_fraction = 0.35
        omega = 2.0 * np.pi / orbital_period_s
        for i, t_i in enumerate(t_np):
            sun_vec = np.array([np.cos(omega * t_i), np.sin(omega * t_i), 0.0])
            pos = (t_i % orbital_period_s) / orbital_period_s
            d = abs(pos - 0.5) - (eclipse_fraction / 2.0)
            eclipse = 0.5 * (1.0 + np.tanh(-d * 50.0))
            from ._thermal_solver import _dT_dt as solver_dT
            out_np[i] = solver_dT(T_np[i], sun_vec, eclipse, self.bus)
        return torch.from_numpy(out_np).to(T.device).to(T.dtype)

    # ── Training step: hybrid loss ──────────────────────────────────────

    def compute_loss(
        self,
        observed_t: torch.Tensor,
        observed_T: torch.Tensor,
    ) -> torch.Tensor:
        """Hybrid PINN loss: data fit + physics residual at collocation points.

        observed_t: (N,) times in seconds
        observed_T: (N, n_zones) ground-truth temperatures
        """
        # Data loss
        pred = self.forward(observed_t)
        data_loss = ((pred - observed_T) ** 2).mean()

        # Physics residual at sampled collocation points
        device = next(self.parameters()).device
        t_coll = sample_collocation_times(
            self.config.n_collocation,
            t_min_s=0.0,
            t_max_s=self.config.horizon_s,
            device=device,
        )
        residual = self._physics_residual_at(t_coll)

        loss_cfg = HybridLossConfig(
            w_data=self.config.w_data,
            w_physics=self.config.w_physics,
            n_collocation_points=self.config.n_collocation,
            collocation_t_max_s=self.config.horizon_s,
        )
        return hybrid_pinn_loss(
            data_loss=data_loss,
            physics_residual=residual,
            cfg=loss_cfg,
        )


# ── PhysicsModel adapter so this plugs into PINNResidualDetector ───────


class ThermalPinnPhysicsModel:
    """Wraps a trained ThermalPinn as a `PhysicsModel`.

    `predict()` evaluates the PINN at the window's timestamps to get a
    per-step temperature prediction for each thermal channel.
    """

    name = "thermal_pinn"

    def __init__(
        self,
        pinn: ThermalPinn,
        channel_to_zone: dict[str, int],
    ) -> None:
        self._pinn = pinn
        self._channel_to_zone = channel_to_zone

    def covered_channels(self) -> set[str]:
        return set(self._channel_to_zone)

    def predict(self, window: TelemetryWindow) -> np.ndarray:
        ts_ms = window.timestamps.astype(np.float64)
        t_s = (ts_ms - ts_ms[0]) / 1000.0
        device = next(self._pinn.parameters()).device
        t_torch = torch.from_numpy(t_s).float().to(device)
        self._pinn.eval()
        with torch.no_grad():
            T_pred = self._pinn(t_torch).cpu().numpy()    # (T, n_zones)
        pred = np.zeros_like(window.tensor)
        for i, ch in enumerate(window.channels):
            zone = self._channel_to_zone.get(ch.name)
            if zone is not None and 0 <= zone < T_pred.shape[1]:
                pred[:, i] = T_pred[:, zone].astype(np.float32)
        return pred


def make_thermal_pinn_physics(
    pinn: ThermalPinn, channel_to_zone: dict[str, int]
) -> ThermalPinnPhysicsModel:
    """Convenience: build the PhysicsModel adapter."""
    return ThermalPinnPhysicsModel(pinn=pinn, channel_to_zone=channel_to_zone)
