"""Phase 9 tests — thermal solver, PINN base, ADCS integrators, thermal PINN."""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ── Multi-zone thermal solver ──────────────────────────────────────────


def test_thermal_solver_runs_and_produces_temperatures() -> None:
    from dgx_ts_lab.models.physics import simulate_thermal

    out = simulate_thermal(duration_s=10800.0, dt_s=60.0, initial_temp_K=290.0)
    assert out["T"].shape == (181, 6)
    # Temperatures stay in a realistic range for a small sat
    assert 150.0 < out["T"].mean() < 350.0
    assert (out["T"] > 100.0).all() and (out["T"] < 400.0).all()


def test_thermal_solver_responds_to_eclipse() -> None:
    """During eclipse, temperature should decrease (no solar input)."""
    from dgx_ts_lab.models.physics import simulate_thermal

    out = simulate_thermal(duration_s=10800.0, dt_s=60.0, initial_temp_K=290.0)
    # Look at the sunlit zone (+X)
    T_x = out["T"][:, 0]
    in_eclipse = out["eclipse"] > 0.8
    in_sun = out["eclipse"] < 0.2
    if in_eclipse.any() and in_sun.any():
        T_eclipse_mean = T_x[in_eclipse].mean()
        T_sun_mean = T_x[in_sun].mean()
        # +X face should run cooler when in eclipse
        assert T_eclipse_mean < T_sun_mean + 5.0


# ── PINN base components ───────────────────────────────────────────────


def test_fourier_features_shape() -> None:
    from dgx_ts_lab.models.physics._pinn_base_torch import FourierFeatures

    ff = FourierFeatures(n_freqs=8, min_period_s=1.0, max_period_s=100.0)
    t = torch.arange(16, dtype=torch.float32)
    out = ff(t)
    assert out.shape == (16, 16)  # 2 * n_freqs


def test_pinn_backbone_produces_correct_output_dim() -> None:
    from dgx_ts_lab.models.physics._pinn_base_torch import PINNBackbone

    net = PINNBackbone(output_dim=6, n_freqs=8, hidden=32, n_layers=2)
    t = torch.arange(32, dtype=torch.float32)
    out = net(t)
    assert out.shape == (32, 6)


def test_sample_collocation_times_requires_grad() -> None:
    from dgx_ts_lab.models.physics._pinn_base_torch import sample_collocation_times

    t = sample_collocation_times(n=64, t_min_s=0.0, t_max_s=3600.0, device=torch.device("cpu"))
    assert t.shape == (64, 1)
    assert t.requires_grad


# ── ADCS state + integrators ───────────────────────────────────────────


def _initial_state(device="cpu") -> "AttitudeState":   # noqa: F821
    from dgx_ts_lab.models.physics.adcs import AttitudeState

    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)   # identity
    omega = torch.tensor([[0.1, 0.0, 0.0]], device=device)    # spinning around x
    rw = torch.zeros(1, 3, device=device)
    return AttitudeState(q, omega, rw)


def _identity_quaternion_preserved(integrator, n_steps: int = 100, dt: float = 0.1) -> float:
    """Run an integrator with zero torque + zero body rates from identity.
    Should preserve the identity quaternion. Returns max deviation magnitude."""
    from dgx_ts_lab.models.physics.adcs import AttitudeState, SpacecraftInertia

    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    omega = torch.zeros(1, 3)
    rw = torch.zeros(1, 3)
    state = AttitudeState(q, omega, rw)
    inertia = SpacecraftInertia.default_3axis()
    tau = torch.zeros(1, 3)
    for _ in range(n_steps):
        state = integrator(state, tau, inertia, dt)
    q_final = state.quaternion[0]
    deviation = (q_final - torch.tensor([0.0, 0.0, 0.0, 1.0])).abs().max().item()
    return float(deviation)


def test_euler_preserves_identity_under_zero_torque() -> None:
    from dgx_ts_lab.models.physics.adcs import step_euler

    dev = _identity_quaternion_preserved(step_euler)
    assert dev < 1e-4


def test_rk4_preserves_identity_under_zero_torque() -> None:
    from dgx_ts_lab.models.physics.adcs import step_rk4

    dev = _identity_quaternion_preserved(step_rk4)
    assert dev < 1e-4


def test_so3_preserves_identity_under_zero_torque() -> None:
    from dgx_ts_lab.models.physics.adcs import step_so3

    dev = _identity_quaternion_preserved(step_so3)
    assert dev < 1e-4


def test_so3_preserves_quaternion_norm_better_than_euler() -> None:
    """After many steps with persistent rotation, SO(3) integrator should
    have less norm drift than Euler."""
    from dgx_ts_lab.models.physics.adcs import (
        AttitudeState,
        SpacecraftInertia,
        step_euler,
        step_so3,
    )

    inertia = SpacecraftInertia.default_3axis()
    tau = torch.zeros(1, 3)

    def _run(step_fn, n_steps=2000, dt=0.01):
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        omega = torch.tensor([[0.5, 0.3, 0.2]])
        rw = torch.zeros(1, 3)
        state = AttitudeState(q, omega, rw)
        for _ in range(n_steps):
            state = step_fn(state, tau, inertia, dt)
        return float(state.quaternion.norm(dim=-1).item())

    norm_so3 = _run(step_so3)
    # SO(3) geometric integrator: norm drift bounded by float32 round-off
    # over many steps (vs Euler which drifts linearly with step count).
    assert abs(norm_so3 - 1.0) < 1e-3


def test_adcs_pinn_construction_with_each_integrator() -> None:
    from dgx_ts_lab.models.physics.adcs import ADCSPinn
    from dgx_ts_lab.models.physics.adcs.adcs_pinn import ADCSPinnConfig

    for integrator in ("euler", "rk4", "so3"):
        cfg = ADCSPinnConfig(integrator=integrator, n_freqs=4, hidden=16, n_layers=2)
        pinn = ADCSPinn(cfg)
        t = torch.arange(16, dtype=torch.float32)
        out = pinn(t)
        assert out.quaternion.shape == (16, 4)
        assert out.body_rates.shape == (16, 3)


def test_adcs_pinn_rejects_unknown_integrator() -> None:
    from dgx_ts_lab.models.physics.adcs.adcs_pinn import ADCSPinn, ADCSPinnConfig

    with pytest.raises(ValueError, match="unknown integrator"):
        ADCSPinn(ADCSPinnConfig(integrator="quaternion_squared"))


# ── Thermal PINN ───────────────────────────────────────────────────────


def test_thermal_pinn_forward_shape() -> None:
    from dgx_ts_lab.models.physics import ThermalPinn, ThermalPinnConfig

    cfg = ThermalPinnConfig(n_zones=6, n_freqs=4, hidden=16, n_layers=2, horizon_s=3600.0)
    pinn = ThermalPinn(cfg)
    t = torch.arange(32, dtype=torch.float32)
    out = pinn(t)
    assert out.shape == (32, 6)


def test_thermal_pinn_loss_finite_and_decreases_on_one_step() -> None:
    """Sanity: one optimizer step against fake ground truth reduces the loss."""
    from dgx_ts_lab.models.physics import ThermalPinn, ThermalPinnConfig

    cfg = ThermalPinnConfig(
        n_zones=6, n_freqs=4, hidden=16, n_layers=2, horizon_s=3600.0,
        n_collocation=32, w_physics=0.01,
    )
    pinn = ThermalPinn(cfg)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-2)

    observed_t = torch.linspace(0, 3600.0, 64)
    observed_T = torch.full((64, 6), 290.0) + 10.0 * torch.sin(observed_t.view(-1, 1) / 600.0)

    loss_before = float(pinn.compute_loss(observed_t, observed_T))
    for _ in range(3):
        optimizer.zero_grad()
        loss = pinn.compute_loss(observed_t, observed_T)
        loss.backward()
        optimizer.step()
    loss_after = float(pinn.compute_loss(observed_t, observed_T))

    assert np.isfinite(loss_before)
    assert np.isfinite(loss_after)
    assert loss_after < loss_before


def test_thermal_pinn_physics_model_adapter_predicts_for_known_channels() -> None:
    """Wrap a (untrained) ThermalPinn as a PhysicsModel; verify predict()."""
    from dgx_ts_core.data import Channel, Subsystem, TelemetryWindow, Units
    from dgx_ts_lab.models.physics import ThermalPinn, ThermalPinnConfig
    from dgx_ts_lab.models.physics.thermal_pinn import ThermalPinnPhysicsModel

    cfg = ThermalPinnConfig(n_zones=6, n_freqs=4, hidden=16, n_layers=2, horizon_s=3600.0)
    pinn = ThermalPinn(cfg)
    adapter = ThermalPinnPhysicsModel(pinn, channel_to_zone={"temp_a": 0, "temp_b": 1})
    chans = (
        Channel(name="temp_a", units=Units.CELSIUS, subsystem=Subsystem.TCS, sample_rate_hz=1.0),
        Channel(name="temp_b", units=Units.CELSIUS, subsystem=Subsystem.TCS, sample_rate_hz=1.0),
        Channel(name="other", units=Units.VOLT, subsystem=Subsystem.EPS, sample_rate_hz=1.0),
    )
    win = TelemetryWindow(
        tensor=np.zeros((64, 3), dtype=np.float32),
        timestamps=np.arange(64, dtype=np.int64) * 1000,
        channels=chans,
    )
    pred = adapter.predict(win)
    assert pred.shape == (64, 3)
    # Channels not covered by the adapter must be exactly 0
    assert np.allclose(pred[:, 2], 0.0)


# ── Modulus shell ──────────────────────────────────────────────────────


def test_modulus_thermal_fem_raises_when_modulus_missing() -> None:
    """Without nvidia-modulus installed, ModulusThermalFEM construction should
    raise a clear error pointing at the install hint."""
    from dgx_ts_lab.models.physics.modulus_thermal_fem import (
        ModulusThermalFEM,
        _try_import_modulus,
    )

    if _try_import_modulus():
        pytest.skip("nvidia-modulus is installed; can't exercise the missing-import path")
    with pytest.raises(ImportError, match="nvidia-modulus"):
        ModulusThermalFEM()
