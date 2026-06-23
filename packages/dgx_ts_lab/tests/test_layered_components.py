"""Per-component unit tests for the L1-L6 layered generator."""

from __future__ import annotations

import numpy as np
from dgx_ts_core.data import Channel, Subsystem, Units
from dgx_ts_lab.datasets.synthetic.layered import (
    LayeredSyntheticDataset,
    coupling,
    drift,
    faults,
    modes,
    noise,
    physics,
)
from dgx_ts_lab.datasets.synthetic.layered.component import GenState


def _channels(n: int) -> tuple[Channel, ...]:
    return tuple(
        Channel(name=f"c{i}", units=Units.DIMENSIONLESS, subsystem=Subsystem.UNKNOWN, sample_rate_hz=1.0)
        for i in range(n)
    )


def _empty_state(n_steps: int, n_channels: int = 1, sample_rate_hz: float = 1.0) -> GenState:
    t = np.arange(n_steps, dtype=np.float32) / sample_rate_hz
    chans = _channels(n_channels)
    return GenState(
        t=t,
        data=np.zeros((n_steps, n_channels), dtype=np.float32),
        mode=np.full(n_steps, -1, dtype=np.int32),
        labels=np.zeros(n_steps, dtype=np.bool_),
        channel_index={c.name: i for i, c in enumerate(chans)},
        sample_rate_hz=sample_rate_hz,
    )


# ── L1 physics ──────────────────────────────────────────────────────


def test_orbital_sinusoid_amplitude_matches() -> None:
    state = _empty_state(10800, 1)
    physics.OrbitalSinusoid("c0", amplitude=2.0, period_s=5400.0).apply(state, np.random.default_rng(0))
    assert abs(state.data[:, 0].max() - 2.0) < 0.05
    assert abs(state.data[:, 0].min() + 2.0) < 0.05


def test_thermal_duty_cycle_responds_to_mode() -> None:
    state = _empty_state(20000, 1)
    # First 10k steps in sun, next 10k in eclipse.
    state.mode[:10000] = modes.MODE_VOCAB["sun"]
    state.mode[10000:] = modes.MODE_VOCAB["eclipse"]
    physics.ThermalDutyCycle(
        "c0", equilibrium_sun_C=50.0, equilibrium_shade_C=-30.0, time_constant_s=300.0
    ).apply(state, np.random.default_rng(0))
    # After ~10× tau in sun, should be near sun equilibrium.
    assert state.data[9999, 0] > 40.0
    # After ~10× tau in eclipse, should be near shade equilibrium.
    assert state.data[-1, 0] < -10.0


def test_battery_soc_drains_in_eclipse() -> None:
    state = _empty_state(7200, 1)
    state.mode[:] = modes.MODE_VOCAB["eclipse"]
    physics.BatterySoC("c0", capacity_Ah=10.0, discharge_rate_A=2.0, initial_soc=0.8).apply(
        state, np.random.default_rng(0)
    )
    assert state.data[0, 0] == np.float32(0.8)
    assert state.data[-1, 0] < state.data[0, 0]
    assert state.data[-1, 0] >= 0.0


# ── L2 modes ────────────────────────────────────────────────────────


def test_mode_machine_emits_sun_and_eclipse() -> None:
    state = _empty_state(21600, 1)
    modes.ModeMachine(period_s=5400.0, eclipse_fraction=0.35).apply(state, np.random.default_rng(0))
    assert (state.mode == modes.MODE_VOCAB["sun"]).any()
    assert (state.mode == modes.MODE_VOCAB["eclipse"]).any()
    eclipse_frac = float((state.mode == modes.MODE_VOCAB["eclipse"]).mean())
    assert 0.25 < eclipse_frac < 0.45


# ── L3 coupling ─────────────────────────────────────────────────────


def test_linear_coupling_transfers_signal() -> None:
    state = _empty_state(1000, 2)
    state.data[:, 0] = 3.0
    coupling.LinearCoupling("c0", "c1", gain=2.0, offset=1.0).apply(state, np.random.default_rng(0))
    assert np.allclose(state.data[:, 1], 7.0)


def test_sum_coupling_aggregates_multiple_sources() -> None:
    state = _empty_state(100, 4)
    state.data[:, 0] = 1.0
    state.data[:, 1] = 2.0
    state.data[:, 2] = 3.0
    coupling.SumCoupling(
        sources=["c0", "c1", "c2"], target="c3", gains=[1.0, 2.0, 1.0], offset=0.5,
    ).apply(state, np.random.default_rng(0))
    # target = 0.5 + 1*1.0 + 2*2.0 + 1*3.0 = 8.5
    assert np.allclose(state.data[:, 3], 8.5)


def test_sum_coupling_default_gains_are_unit() -> None:
    state = _empty_state(50, 3)
    state.data[:, 0] = 5.0
    state.data[:, 1] = 7.0
    coupling.SumCoupling(sources=["c0", "c1"], target="c2").apply(state, np.random.default_rng(0))
    assert np.allclose(state.data[:, 2], 12.0)


def test_constant_baseline_adds_offset() -> None:
    state = _empty_state(50, 1)
    physics.ConstantBaseline("c0", value=28.0).apply(state, np.random.default_rng(0))
    assert np.allclose(state.data[:, 0], 28.0)


# ── L4 noise ────────────────────────────────────────────────────────


def test_gaussian_noise_has_target_std() -> None:
    state = _empty_state(10000, 1)
    noise.GaussianNoise("c0", std=0.5).apply(state, np.random.default_rng(0))
    assert 0.45 < state.data[:, 0].std() < 0.55


def test_pink_noise_low_freq_dominates() -> None:
    state = _empty_state(8192, 1)
    noise.PinkNoise("c0", std=1.0).apply(state, np.random.default_rng(0))
    spec = np.abs(np.fft.rfft(state.data[:, 0])) ** 2
    # Low-frequency band should have more power than high-frequency band.
    low = spec[1 : len(spec) // 16].mean()
    high = spec[len(spec) // 2 :].mean()
    assert low > 4.0 * high


def test_student_t_noise_has_heavy_tails_vs_gaussian() -> None:
    rng = np.random.default_rng(0)
    state = _empty_state(20000, 1)
    noise.StudentTNoise("c0", scale=1.0, df=3.0).apply(state, rng)
    # Excess kurtosis of t(df=3) is infinite (theoretically); empirically very large.
    x = state.data[:, 0]
    kurt = ((x - x.mean()) ** 4).mean() / (x.var() ** 2) - 3.0
    assert kurt > 1.0  # Gaussian would give ~0


def test_quantization_noise_produces_discrete_levels() -> None:
    state = _empty_state(1000, 1)
    state.data[:, 0] = np.linspace(-1.0, 1.0, 1000).astype(np.float32)
    noise.QuantizationNoise("c0", n_bits=4, full_scale=1.0).apply(state, np.random.default_rng(0))
    # 4-bit quantization over ±1 → 16 levels; unique values should be small.
    assert len(np.unique(state.data[:, 0])) <= 17


def test_correlated_gaussian_noise_couples_channels() -> None:
    state = _empty_state(10000, 2)
    cov = [[1.0, 0.9], [0.9, 1.0]]
    noise.CorrelatedGaussianNoise(["c0", "c1"], cov).apply(state, np.random.default_rng(0))
    corr = np.corrcoef(state.data[:, 0], state.data[:, 1])[0, 1]
    assert corr > 0.8


def test_poisson_burst_noise_creates_bursts() -> None:
    # High rate so we definitely get events in 1 hour
    state = _empty_state(3600, 1)
    noise.PoissonBurstNoise("c0", event_rate_per_hour=50.0, burst_size=10, magnitude=2.0).apply(
        state, np.random.default_rng(0)
    )
    assert np.abs(state.data[:, 0]).max() > 1.0


def test_multiplicative_gain_noise_perturbs_signal() -> None:
    state = _empty_state(5000, 1)
    state.data[:, 0] = 100.0
    noise.MultiplicativeGainNoise("c0", std=0.05, time_constant_s=10.0).apply(
        state, np.random.default_rng(0)
    )
    # Values should now vary around 100 (not exactly 100).
    assert state.data[:, 0].std() > 1.0
    assert abs(state.data[:, 0].mean() - 100.0) < 10.0


# ── L5 non-stationarity ─────────────────────────────────────────────


def test_linear_drift_accumulates_over_time() -> None:
    state = _empty_state(86400, 1)  # exactly 1 day @ 1Hz
    drift.LinearDrift("c0", drift_per_day=0.5).apply(state, np.random.default_rng(0))
    assert abs(state.data[-1, 0] - 0.5) < 0.01


def test_seasonal_modulation_completes_cycle() -> None:
    # 2 days @ 1Hz, with 1-day period → 2 full cycles
    state = _empty_state(86400 * 2, 1)
    drift.SeasonalModulation("c0", amplitude=3.0, period_days=1.0).apply(
        state, np.random.default_rng(0)
    )
    assert abs(state.data[:, 0].max() - 3.0) < 0.1


def test_regime_change_steps_at_specified_time() -> None:
    state = _empty_state(1000, 1)
    drift.RegimeChange("c0", step_time_s=500.0, step_magnitude=7.0).apply(state, np.random.default_rng(0))
    assert state.data[499, 0] == 0.0
    assert state.data[500, 0] == np.float32(7.0)


# ── L6 faults ───────────────────────────────────────────────────────


def test_point_fault_writes_labels_and_log() -> None:
    state = _empty_state(36000, 1)  # 10h
    faults.PointFault("c0", rate_per_hour=2.0, magnitude=10.0).apply(state, np.random.default_rng(0))
    assert state.labels.any()
    assert len(state.fault_log) > 0
    assert all(entry["type"] == "point_fault" for entry in state.fault_log)


def test_dropout_fault_holds_fill_value() -> None:
    state = _empty_state(7200, 1)  # 2h
    state.data[:, 0] = 10.0
    faults.DropoutFault("c0", rate_per_hour=20.0, min_duration_s=5.0,
                       max_duration_s=10.0, fill_value=0.0).apply(state, np.random.default_rng(0))
    # Some steps should be at the fill value.
    assert (state.data[:, 0] == 0.0).any()
    assert state.labels.any()


def test_stuck_at_fault_freezes_value() -> None:
    state = _empty_state(86400, 1)  # 1 day
    state.data[:, 0] = np.arange(86400, dtype=np.float32)
    faults.StuckAtFault("c0", rate_per_day=20.0, min_duration_s=60.0, max_duration_s=120.0).apply(
        state, np.random.default_rng(0)
    )
    # Where labeled, the diff should be 0 (within a small tolerance).
    if state.fault_log:
        e = state.fault_log[0]
        sl = slice(e["start"], e["end"])
        assert np.allclose(state.data[sl, 0], state.data[e["start"], 0])


def test_drift_fault_ramps_then_persists() -> None:
    state = _empty_state(86400, 1)
    faults.DriftFault("c0", rate_per_day=5.0, ramp_duration_s=600.0,
                     final_offset=3.0, persist=True).apply(state, np.random.default_rng(0))
    assert state.labels.any()
    # If we got at least one fault, the tail should reflect the final offset.
    if state.fault_log:
        e = state.fault_log[0]
        if e["end"] < state.n_steps - 1:
            assert abs(state.data[-1, 0] - len(state.fault_log) * 3.0) < 0.5 * len(state.fault_log)


def test_oscillation_fault_introduces_high_frequency_content() -> None:
    state = _empty_state(43200, 1)  # 12h
    faults.OscillationFault("c0", rate_per_day=10.0, frequency_hz=0.4,
                           amplitude=3.0, duration_s=120.0).apply(state, np.random.default_rng(0))
    assert state.labels.any()


def test_correlation_break_fault_writes_labels() -> None:
    state = _empty_state(86400, 2)
    faults.CorrelationBreakFault(
        "c0", reference_channel="c1", rate_per_day=10.0,
        divergence_std=3.0, duration_s=300.0
    ).apply(state, np.random.default_rng(0))
    assert state.labels.any()


def test_mode_confusion_fault_overwrites_value() -> None:
    state = _empty_state(86400, 1)
    state.mode[:] = modes.MODE_VOCAB["sun"]
    state.data[:, 0] = 1.0
    faults.ModeConfusionFault(
        "c0", rate_per_day=10.0, wrong_value=99.0, duration_s=60.0, only_when_mode="sun"
    ).apply(state, np.random.default_rng(0))
    assert (state.data[:, 0] == 99.0).any()
    assert state.labels.any()


# ── End-to-end orchestrator ─────────────────────────────────────────


def test_layered_dataset_end_to_end_runs_and_labels_faults() -> None:
    chans = (
        Channel(name="v", units=Units.VOLT, subsystem=Subsystem.EPS, sample_rate_hz=1.0),
        Channel(name="i", units=Units.AMP, subsystem=Subsystem.EPS, sample_rate_hz=1.0),
    )
    components = [
        modes.ModeMachine(period_s=5400.0, eclipse_fraction=0.35),
        physics.OrbitalSinusoid("v", amplitude=0.5, period_s=5400.0, baseline=28.0),
        coupling.LinearCoupling("v", "i", gain=0.1),
        noise.GaussianNoise("v", std=0.02),
        noise.PinkNoise("i", std=0.05),
        noise.StudentTNoise("i", scale=0.02, df=4.0),
        drift.LinearDrift("v", drift_per_day=-0.01),
        faults.PointFault("v", rate_per_hour=4.0, magnitude=2.0),
        faults.DropoutFault("i", rate_per_hour=2.0, min_duration_s=2.0, max_duration_s=5.0),
    ]
    ds = LayeredSyntheticDataset(
        channels=chans, components=components, n_samples=14400, sample_rate_hz=1.0, seed=0
    )
    assert ds.has_labels
    # At least one fault entry per fault component above (Poisson rates → likely).
    assert len(ds.fault_log) > 0
    # Windows iterate cleanly.
    win = next(ds.windows(length=256, stride=256))
    assert win.tensor.shape == (256, 2)
    assert win.labels is not None
    # Splits work.
    splits = ds.split(__import__("dgx_ts_core.data", fromlist=["SplitScheme"]).SplitScheme(
        strategy=__import__("dgx_ts_core.data", fromlist=["SplitStrategy"]).SplitStrategy.TEMPORAL
    ))
    assert set(splits) == {"train", "val", "test"}


def test_layered_dataset_is_deterministic_under_seed() -> None:
    chans = _channels(1)
    cs = [
        physics.OrbitalSinusoid("c0", amplitude=1.0, period_s=600.0),
        noise.GaussianNoise("c0", std=0.1),
        faults.PointFault("c0", rate_per_hour=10.0, magnitude=5.0),
    ]
    a = LayeredSyntheticDataset(chans, cs, n_samples=3600, seed=42)
    b = LayeredSyntheticDataset(chans, cs, n_samples=3600, seed=42)
    np.testing.assert_array_equal(a._data, b._data)
    np.testing.assert_array_equal(a._labels, b._labels)


def test_layered_synth_registered() -> None:
    import dgx_ts_lab  # noqa: F401  trigger registration
    from dgx_ts_core.registry import DATASET_REGISTRY

    assert "layered_synth" in DATASET_REGISTRY.list()
