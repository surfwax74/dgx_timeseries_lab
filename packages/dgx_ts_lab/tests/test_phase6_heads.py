"""Phase 6 tests — task heads, multi-task wrapper, label generation."""

from __future__ import annotations

import numpy as np
import torch
from dgx_ts_core.models import FitMode


def _tiny_layered_dataset(n_samples: int = 2000, emit_multitask: bool = True):
    """Build a tiny layered dataset with multi-task labels."""
    from dgx_ts_core.data import Channel, Subsystem, Units
    from dgx_ts_lab.datasets.synthetic.layered import (
        LayeredSyntheticDataset,
        faults,
        modes,
        noise,
        physics,
    )

    channels = (
        Channel(name="v", units=Units.VOLT, subsystem=Subsystem.EPS, sample_rate_hz=1.0),
        Channel(name="i", units=Units.AMP, subsystem=Subsystem.EPS, sample_rate_hz=1.0),
        Channel(name="t", units=Units.CELSIUS, subsystem=Subsystem.TCS, sample_rate_hz=1.0),
    )
    components = [
        modes.ModeMachine(period_s=600.0, eclipse_fraction=0.35),
        physics.OrbitalSinusoid("v", amplitude=0.5, period_s=600.0, baseline=28.0),
        physics.OrbitalSinusoid("i", amplitude=0.1, period_s=600.0, baseline=1.0),
        physics.ConstantBaseline("t", value=20.0),
        noise.GaussianNoise("v", std=0.02),
        noise.GaussianNoise("i", std=0.02),
        # Lots of faults to make sure aux_labels actually fire on small windows
        faults.PointFault("v", rate_per_hour=20.0, magnitude=5.0),
        faults.DropoutFault("i", rate_per_hour=10.0, min_duration_s=2.0, max_duration_s=5.0),
        faults.StuckAtFault("t", rate_per_day=200.0, min_duration_s=30.0, max_duration_s=60.0),
    ]
    return LayeredSyntheticDataset(
        channels=channels,
        components=components,
        n_samples=n_samples,
        sample_rate_hz=1.0,
        seed=0,
        emit_multitask_labels=emit_multitask,
        next_mode_horizon_s=30.0,
    )


# ── Label generator ─────────────────────────────────────────────────────


def test_multitask_label_computer_basic_shape() -> None:
    from dgx_ts_lab.datasets.synthetic.layered.labels import (
        FAULT_CLASS_INDEX,
        NO_FAULT_CEILING_SECONDS,
        MultiTaskLabelComputer,
    )

    fault_log = [
        {"type": "point_fault", "channel": "v", "start": 100, "end": 101},
        {"type": "drift_fault", "channel": "i", "start": 500, "end": 600},
    ]
    mode_trace = np.zeros(1000, dtype=np.int32)
    mode_trace[200:400] = 1  # eclipse window

    lc = MultiTaskLabelComputer(fault_log, mode_trace, sample_rate_hz=1.0, next_mode_horizon_s=10.0)

    # Class arrays
    assert lc.fault_type_full.shape == (1000,)
    assert lc.fault_type_full[100] == FAULT_CLASS_INDEX["point_fault"]
    assert lc.fault_type_full[550] == FAULT_CLASS_INDEX["drift_fault"]
    assert lc.fault_type_full[0] == 0   # no fault

    # RUL: at step 50, next fault is at 100 → RUL = 50 sec
    assert abs(lc.rul_full[50] - 50.0) < 0.001
    # RUL: after the last fault onset, RUL should be at ceiling
    assert lc.rul_full[700] == NO_FAULT_CEILING_SECONDS

    # Next-mode: at step 195, horizon 10 → mode at 205 should be 1 (eclipse)
    assert lc.next_mode_full[195] == 1
    # Out-of-range gets -1
    assert lc.next_mode_full[-1] == -1


def test_layered_dataset_emits_aux_labels_when_enabled() -> None:
    ds = _tiny_layered_dataset(n_samples=2000, emit_multitask=True)
    win = next(ds.windows(length=128, stride=128))
    assert win.aux_labels is not None
    assert set(win.aux_labels.keys()) == {"fault_type", "rul", "next_mode"}
    assert win.aux_labels["fault_type"].shape == (128,)
    assert win.aux_labels["rul"].shape == (128,)
    assert win.aux_labels["next_mode"].shape == (128,)


def test_layered_dataset_omits_aux_labels_by_default() -> None:
    ds = _tiny_layered_dataset(n_samples=1000, emit_multitask=False)
    win = next(ds.windows(length=128, stride=128))
    assert win.aux_labels is None


# ── HEAD_REGISTRY + Capabilities ────────────────────────────────────────


def test_head_registry_has_all_three_heads() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import HEAD_REGISTRY

    keys = HEAD_REGISTRY.list()
    assert "fault_classifier" in keys
    assert "rul_regressor" in keys
    assert "mode_predictor" in keys


def test_capabilities_supports_multi_task_flag_exists() -> None:
    from dgx_ts_core.models import Capabilities, OutputKind

    caps = Capabilities(
        requires_pretraining=True,
        supports_streaming=False,
        supports_multivariate=True,
        native_context_len=128,
        output_kind=OutputKind.PER_STEP,
        supports_multi_task=True,
    )
    assert caps.supports_multi_task is True


# ── Sat-TSFM encode_pooled_steps ────────────────────────────────────────


def test_sat_tsfm_encode_pooled_steps_returns_correct_shape() -> None:
    from dgx_ts_lab.models.from_scratch.sat_tsfm import SatTSFMModule

    mod = SatTSFMModule(
        max_channels=16, window_length=64, patch_len=16,
        d_model=32, n_layers=1, n_heads=2,
    )
    x = torch.randn(2, 64, 4, dtype=torch.float32)
    out = mod.encode_pooled_steps(x)
    assert out.shape == (2, 64, 32)


# ── Individual head forward + loss ─────────────────────────────────────


def _fake_encoded_and_batch(B: int = 2, T: int = 32, D: int = 32):
    encoded = torch.randn(B, T, D)
    aux = {
        "fault_type": torch.randint(0, 8, (B, T), dtype=torch.long),
        "rul": torch.rand(B, T, dtype=torch.float32) * 100.0,
        "next_mode": torch.randint(0, 6, (B, T), dtype=torch.long),
    }
    return encoded, {"aux_labels": aux}


def test_fault_classifier_loss_and_metrics() -> None:
    from dgx_ts_lab.models.heads import FaultClassifierHead

    head = FaultClassifierHead(d_model=32, num_classes=8)
    encoded, batch = _fake_encoded_and_batch()
    loss = head.compute_loss(encoded, batch)
    assert loss.dim() == 0 and torch.isfinite(loss)
    metrics = head.compute_metrics(encoded, batch)
    assert "fault_classifier.acc" in metrics


def test_rul_regressor_loss_and_metrics() -> None:
    from dgx_ts_lab.models.heads import RULRegressorHead

    head = RULRegressorHead(d_model=32)
    encoded, batch = _fake_encoded_and_batch()
    loss = head.compute_loss(encoded, batch)
    assert loss.dim() == 0 and torch.isfinite(loss)
    metrics = head.compute_metrics(encoded, batch)
    assert "rul_regressor.mae_log_s" in metrics


def test_mode_predictor_loss_and_metrics() -> None:
    from dgx_ts_lab.models.heads import ModePredictorHead

    head = ModePredictorHead(d_model=32, num_modes=6)
    encoded, batch = _fake_encoded_and_batch()
    loss = head.compute_loss(encoded, batch)
    assert loss.dim() == 0 and torch.isfinite(loss)


# ── Multi-task wrapper end-to-end ──────────────────────────────────────


def test_multitask_detector_fit_and_loss() -> None:
    from dgx_ts_lab.models.from_scratch import SatTSFMMultiTaskDetector

    det = SatTSFMMultiTaskDetector(
        window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2,
        heads=[
            {"key": "fault_classifier", "num_classes": 8},
            {"key": "rul_regressor"},
            {"key": "mode_predictor", "num_modes": 6},
        ],
    )
    ds = _tiny_layered_dataset(n_samples=1000, emit_multitask=True)
    det.fit(ds, FitMode.PRETRAIN, {})
    assert det.module is not None
    # Compose a fake batch including aux_labels (as Lightning loop would produce)
    batch = {
        "x": torch.randn(2, 64, 3, dtype=torch.float32),
        "labels": torch.zeros(2, 64, dtype=torch.bool),
        "aux_labels": {
            "fault_type": torch.randint(0, 8, (2, 64), dtype=torch.long),
            "rul": torch.rand(2, 64, dtype=torch.float32) * 100.0,
            "next_mode": torch.randint(0, 6, (2, 64), dtype=torch.long),
        },
    }
    loss = det.compute_loss(batch)
    assert loss.dim() == 0 and torch.isfinite(loss)

    # AD path still works for scoring
    scores = det.compute_score_batch(batch)
    assert scores.shape == (2, 64)


def test_multitask_detector_caps_declare_multi_task() -> None:
    from dgx_ts_lab.models.from_scratch import SatTSFMMultiTaskDetector

    det = SatTSFMMultiTaskDetector(window_length=64, patch_len=16, d_model=32)
    assert det.capabilities.supports_multi_task is True


def test_multitask_detector_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DETECTOR_REGISTRY

    assert "sat_tsfm_multitask" in DETECTOR_REGISTRY.list()


def test_multitask_detector_save_load(tmp_path) -> None:
    from dgx_ts_lab.models.from_scratch import SatTSFMMultiTaskDetector

    det = SatTSFMMultiTaskDetector(
        window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2,
        heads=[{"key": "fault_classifier", "num_classes": 8}],
    )
    ds = _tiny_layered_dataset(n_samples=500, emit_multitask=True)
    det.fit(ds, FitMode.PRETRAIN, {})
    p = tmp_path / "mt.pt"
    det.save(p)
    loaded = SatTSFMMultiTaskDetector.load(p)
    x = torch.randn(1, 64, 3, dtype=torch.float32)
    a = det.compute_score_batch({"x": x})
    b = loaded.compute_score_batch({"x": x})
    torch.testing.assert_close(a, b)


# ── WindowTorchDataset passes aux_labels through ────────────────────────


def test_window_torch_dataset_includes_aux_labels() -> None:
    from dgx_ts_lab.training.window_dataset import WindowTorchDataset

    ds = _tiny_layered_dataset(n_samples=500, emit_multitask=True)
    torch_ds = WindowTorchDataset(ds, length=64, stride=64)
    item = torch_ds[0]
    assert "aux_labels" in item
    assert "fault_type" in item["aux_labels"]
    assert item["aux_labels"]["fault_type"].shape == (64,)
