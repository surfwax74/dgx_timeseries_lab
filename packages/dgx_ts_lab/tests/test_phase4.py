"""Phase 4 tests — Sat-TSFM, PINN wrappers, Subsystem MoE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dgx_ts_core.data import Channel, SplitScheme, SplitStrategy, Subsystem, TelemetryWindow, Units
from dgx_ts_core.models import AnomalyDetector, FitMode, OutputKind


def _tiny_dataset(n_channels: int = 6):
    from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset

    return TrivialSyntheticDataset(
        n_samples=512, n_channels=n_channels, anomaly_rate=0.02,
        anomaly_magnitude_sigmas=8.0, seed=0,
    )


def _tiny_batch(n_channels: int = 6, T: int = 64, B: int = 4):
    return {
        "x": torch.randn(B, T, n_channels, dtype=torch.float32),
        "labels": torch.zeros(B, T, dtype=torch.bool),
    }


# ── Sat-TSFM ─────────────────────────────────────────────────────────────


def test_sat_tsfm_protocol() -> None:
    from dgx_ts_lab.models.from_scratch import SatTSFMDetector

    det = SatTSFMDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    assert isinstance(det, AnomalyDetector)
    assert det.capabilities.requires_pretraining
    assert det.capabilities.output_kind is OutputKind.PER_STEP


def test_sat_tsfm_fit_and_score() -> None:
    from dgx_ts_lab.models.from_scratch import SatTSFMDetector

    det = SatTSFMDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(n_channels=4), FitMode.PRETRAIN, {})
    loss = det.compute_loss(_tiny_batch(n_channels=4, T=64, B=3))
    assert torch.isfinite(loss)
    scores = det.compute_score_batch(_tiny_batch(n_channels=4, T=64, B=3))
    assert scores.shape == (3, 64)


def test_sat_tsfm_channel_flexible() -> None:
    """Same Sat-TSFM should work on 3-channel and 5-channel datasets."""
    from dgx_ts_lab.models.from_scratch import SatTSFMDetector

    det = SatTSFMDetector(
        window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2, max_channels=16,
    )
    det.fit(_tiny_dataset(n_channels=3), FitMode.PRETRAIN, {})
    # Score on a different channel count via a window — the underlying module
    # supports any C ≤ max_channels.
    x5 = torch.randn(2, 64, 5, dtype=torch.float32)
    # Need to populate norm buffers for the extra channels
    with torch.no_grad():
        det.module.norm_mean[3:5].copy_(torch.zeros(2))
        det.module.norm_std[3:5].copy_(torch.ones(2))
    scores = det.compute_score_batch({"x": x5, "labels": torch.zeros(2, 64, dtype=torch.bool)})
    assert scores.shape == (2, 64)


def test_sat_tsfm_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DETECTOR_REGISTRY

    assert "sat_tsfm" in DETECTOR_REGISTRY.list()


# ── PINN physics models ──────────────────────────────────────────────────


def _eps_window(T: int = 256, channel_specs: list[str] | None = None) -> TelemetryWindow:
    """Build a window with realistic EPS channel names so the physics models
    can recognize them via name patterns."""
    if channel_specs is None:
        channel_specs = [
            "sa_px_str1_current", "sa_px_current", "sa_px_voltage", "sa_px_sada_angle",
            "sa_px_temp_top", "bat_a_temp1", "pdu_a_temp",
            "bat_a_soc", "bat_a_voltage",
            "unknown_channel",   # physics should leave this at 0
        ]
    channels = tuple(
        Channel(name=n, units=Units.DIMENSIONLESS, subsystem=Subsystem.EPS, sample_rate_hz=1.0)
        for n in channel_specs
    )
    return TelemetryWindow(
        tensor=np.zeros((T, len(channels)), dtype=np.float32),
        timestamps=np.arange(T, dtype=np.int64) * 1000,
        channels=channels,
    )


def test_orbital_residual_covers_solar_channels() -> None:
    from dgx_ts_lab.models.physics import OrbitalResidual

    phys = OrbitalResidual()
    win = _eps_window(T=5400)  # one orbit
    pred = phys.predict(win)
    # Orbital model covers electrical solar channels but NOT thermal (that's ThermalResidual).
    orbital_covered = {"sa_px_str1_current", "sa_px_current", "sa_px_voltage", "sa_px_sada_angle"}
    for i, ch in enumerate(win.channels):
        if ch.name in orbital_covered:
            assert np.abs(pred[:, i]).max() > 0.01, f"{ch.name} should be predicted"
        elif ch.name == "unknown_channel":
            assert np.allclose(pred[:, i], 0.0), "unknown channels must be 0"
        elif "_temp" in ch.name or ch.name.startswith("bat_") or ch.name.startswith("pdu_"):
            # Orbital model's job ends at electrical solar — temps/batteries/PDUs are 0
            assert np.allclose(pred[:, i], 0.0), f"{ch.name} should NOT be predicted by orbital"


def test_thermal_residual_predicts_temperatures() -> None:
    from dgx_ts_lab.models.physics import ThermalResidual

    phys = ThermalResidual()
    win = _eps_window(T=5400 * 2)
    pred = phys.predict(win)
    # Temperature channels should converge to non-zero equilibria
    for i, ch in enumerate(win.channels):
        if "_temp" in ch.name:
            assert pred[-1, i] != 0.0


def test_battery_residual_predicts_soc_and_voltage() -> None:
    from dgx_ts_lab.models.physics import BatteryResidual

    phys = BatteryResidual()
    win = _eps_window(T=5400)
    pred = phys.predict(win)
    soc_idx = [i for i, ch in enumerate(win.channels) if ch.name == "bat_a_soc"][0]
    v_idx = [i for i, ch in enumerate(win.channels) if ch.name == "bat_a_voltage"][0]
    # SoC should be ∈ [0, 1] across the window
    assert 0.0 <= pred[:, soc_idx].min() <= pred[:, soc_idx].max() <= 1.0
    # Voltage should track 27-29.5V range
    assert 26.5 < pred[:, v_idx].mean() < 30.0


def test_pinn_wrapper_composes_correctly() -> None:
    from dgx_ts_lab.models.from_scratch import PatchTSTMAEDetector
    from dgx_ts_lab.models.physics import OrbitalResidual, PINNResidualDetector

    inner = PatchTSTMAEDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    phys = OrbitalResidual()
    wrapped = PINNResidualDetector(inner=inner, physics=phys)
    assert isinstance(wrapped, AnomalyDetector)
    # Same capabilities as inner (caps pass-through)
    assert wrapped.capabilities.requires_pretraining == inner.capabilities.requires_pretraining


def test_pinn_residual_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DETECTOR_REGISTRY

    assert "pinn_residual" in DETECTOR_REGISTRY.list()


# ── Subsystem MoE ────────────────────────────────────────────────────────


def test_subsystem_moe_protocol() -> None:
    from dgx_ts_lab.models.from_scratch import SubsystemMoEDetector

    det = SubsystemMoEDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    assert isinstance(det, AnomalyDetector)


def test_subsystem_moe_routing_by_metadata() -> None:
    """Verify each channel is routed to the expert matching its subsystem."""
    from dgx_ts_lab.models.from_scratch import SubsystemMoEDetector

    det = SubsystemMoEDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(n_channels=3), FitMode.PRETRAIN, {})
    # All channels in trivial_synth are Subsystem.UNKNOWN
    assert det.module is not None
    expected_id = list(Subsystem).index(Subsystem.UNKNOWN)
    assert (det.module.channel_to_subsystem_id == expected_id).all()


def test_subsystem_moe_fit_and_score() -> None:
    from dgx_ts_lab.models.from_scratch import SubsystemMoEDetector

    det = SubsystemMoEDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(n_channels=4), FitMode.PRETRAIN, {})
    loss = det.compute_loss(_tiny_batch(n_channels=4, T=64, B=2))
    assert torch.isfinite(loss)
    scores = det.compute_score_batch(_tiny_batch(n_channels=4, T=64, B=2))
    assert scores.shape == (2, 64)


def test_subsystem_moe_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DETECTOR_REGISTRY

    assert "subsystem_moe" in DETECTOR_REGISTRY.list()


# ── FSDP strategy plumbing ───────────────────────────────────────────────


def test_fsdp_strategy_kwargs_buildable() -> None:
    from dgx_ts_lab.training.strategies.fsdp import build_fsdp_strategy_kwargs

    kwargs = build_fsdp_strategy_kwargs(
        {"fsdp_auto_wrap_min_params": 50_000_000, "fsdp_activation_checkpointing": True}
    )
    # Should produce a dict with at least sharding_strategy + auto_wrap_policy
    assert "sharding_strategy" in kwargs
    assert kwargs["sharding_strategy"] == "FULL_SHARD"
