"""Phase 5 tests — ONNX export round-trip + artifact writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from dgx_ts_core.models import FitMode


def _tiny_dataset(n_channels: int = 3):
    from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset

    return TrivialSyntheticDataset(
        n_samples=400, n_channels=n_channels, anomaly_rate=0.02,
        anomaly_magnitude_sigmas=10.0, seed=0,
    )


# ── Model card + feature schema writers ─────────────────────────────────


def test_model_card_writer_emits_yaml(tmp_path: Path) -> None:
    from dgx_ts_core.models import Capabilities, OutputKind
    from dgx_ts_lab.serving import write_model_card

    caps = Capabilities(
        requires_pretraining=True,
        supports_streaming=False,
        supports_multivariate=True,
        native_context_len=256,
        output_kind=OutputKind.PER_STEP,
        supports_export_onnx=True,
        supports_export_threshold_baked=True,
    )
    out = tmp_path / "model_card.yaml"
    write_model_card(
        detector_name="test_det",
        detector_version="0.1.0",
        capabilities=caps,
        intended_subsystem="eps",
        training_dataset="trivial_synth",
        metrics={"f1": 0.85, "roc_auc": 0.95},
        calibrated_threshold=3.14,
        output_path=out,
    )
    loaded = yaml.safe_load(out.read_text())
    assert loaded["detector_name"] == "test_det"
    assert loaded["calibrated_threshold"] == 3.14
    assert loaded["capabilities"]["output_kind"] == "per_step"
    assert loaded["capabilities"]["supports_export_threshold_baked"] is True
    assert "exported_at" in loaded["extra"]


def test_feature_schema_writer_emits_yaml(tmp_path: Path) -> None:
    from dgx_ts_lab.serving import write_feature_schema

    ds = _tiny_dataset(n_channels=4)
    out = tmp_path / "feature_schema.yaml"
    write_feature_schema(
        channels=ds.channels,
        sample_rate_hz=ds.sample_rate_hz,
        window_length=128,
        stats=ds.stats(),
        output_path=out,
    )
    loaded = yaml.safe_load(out.read_text())
    assert loaded["window_length"] == 128
    assert len(loaded["channels"]) == 4
    assert "means" in loaded["normalization"]
    assert len(loaded["normalization"]["means"]) == 4


# ── ONNX export — PatchTST+MAE ──────────────────────────────────────────


def test_patchtst_onnx_export_writes_both_artifacts(tmp_path: Path) -> None:
    from dgx_ts_lab.models.from_scratch import PatchTSTMAEDetector
    from dgx_ts_lab.serving import export_detector

    det = PatchTSTMAEDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(n_channels=3), FitMode.PRETRAIN, {})
    written = export_detector(
        det, output_dir=tmp_path, threshold=1.0, n_channels=3, window_length=64,
    )
    assert (tmp_path / "model.onnx").exists()
    assert (tmp_path / "model_with_threshold.onnx").exists()
    assert "model" in written
    assert "model_with_threshold" in written


def test_patchtst_onnx_runs_via_onnxruntime(tmp_path: Path) -> None:
    """ONNX export + onnxruntime inference + numeric match to in-process detector."""
    import onnxruntime as ort
    from dgx_ts_lab.models.from_scratch import PatchTSTMAEDetector
    from dgx_ts_lab.serving import export_detector

    det = PatchTSTMAEDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(n_channels=3), FitMode.PRETRAIN, {})
    export_detector(det, output_dir=tmp_path, threshold=1.0, n_channels=3, window_length=64)

    # Load + run ONNX
    sess = ort.InferenceSession(str(tmp_path / "model.onnx"))
    x_np = np.random.randn(2, 64, 3).astype(np.float32)
    onnx_scores = sess.run(["scores"], {"x": x_np})[0]
    assert onnx_scores.shape == (2, 64)

    # Compare to in-process detector
    inproc_scores = (
        det.compute_score_batch({"x": torch.from_numpy(x_np)}).cpu().numpy()
    )
    np.testing.assert_allclose(onnx_scores, inproc_scores, atol=1e-4, rtol=1e-3)


def test_patchtst_threshold_baked_onnx_returns_bool(tmp_path: Path) -> None:
    import onnxruntime as ort
    from dgx_ts_lab.models.from_scratch import PatchTSTMAEDetector
    from dgx_ts_lab.serving import export_detector

    det = PatchTSTMAEDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(n_channels=3), FitMode.PRETRAIN, {})
    export_detector(det, output_dir=tmp_path, threshold=0.5, n_channels=3, window_length=64)

    sess = ort.InferenceSession(str(tmp_path / "model_with_threshold.onnx"))
    x_np = np.random.randn(2, 64, 3).astype(np.float32)
    out = sess.run(["is_anomaly"], {"x": x_np})[0]
    assert out.dtype == bool
    assert out.shape == (2, 64)


# ── ONNX export — Sat-TSFM ──────────────────────────────────────────────


def test_sat_tsfm_onnx_export(tmp_path: Path) -> None:
    import onnxruntime as ort
    from dgx_ts_lab.models.from_scratch import SatTSFMDetector
    from dgx_ts_lab.serving import export_detector

    det = SatTSFMDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(n_channels=4), FitMode.PRETRAIN, {})
    export_detector(det, output_dir=tmp_path, threshold=0.5, n_channels=4, window_length=64)

    sess = ort.InferenceSession(str(tmp_path / "model.onnx"))
    x_np = np.random.randn(1, 64, 4).astype(np.float32)
    onnx_scores = sess.run(["scores"], {"x": x_np})[0]
    assert onnx_scores.shape == (1, 64)


# ── Triton config emitter ───────────────────────────────────────────────


def test_triton_store_layout(tmp_path: Path) -> None:
    from dgx_ts_lab.models.from_scratch import PatchTSTMAEDetector
    from dgx_ts_lab.serving import export_detector, write_triton_ensemble

    det = PatchTSTMAEDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(n_channels=3), FitMode.PRETRAIN, {})
    export_dir = tmp_path / "exported"
    onnx_paths = export_detector(
        det, output_dir=export_dir, threshold=1.0, n_channels=3, window_length=64,
    )
    triton_store = tmp_path / "triton"
    write_triton_ensemble(
        model_name="patchtst_mae",
        onnx_paths=onnx_paths,
        triton_store=triton_store,
        n_channels=3,
        window_length=64,
    )
    # Both raw + baked dirs
    assert (triton_store / "patchtst_mae" / "config.pbtxt").exists()
    assert (triton_store / "patchtst_mae" / "1" / "model.onnx").exists()
    assert (triton_store / "patchtst_mae_with_threshold" / "config.pbtxt").exists()
    # config.pbtxt format sanity
    pbtxt = (triton_store / "patchtst_mae" / "config.pbtxt").read_text()
    assert 'platform: "onnxruntime_onnx"' in pbtxt
    assert "TYPE_FP32" in pbtxt


# ── Export registry ─────────────────────────────────────────────────────


def test_export_registry_finds_factories() -> None:
    from dgx_ts_lab.models.from_scratch import PatchTSTMAEDetector, SatTSFMDetector
    from dgx_ts_lab.serving import EXPORT_REGISTRY

    assert PatchTSTMAEDetector in EXPORT_REGISTRY
    assert SatTSFMDetector in EXPORT_REGISTRY


def test_export_unsupported_detector_raises_clearly() -> None:
    from dgx_ts_lab.models.baseline import RollingMeanDetector
    from dgx_ts_lab.serving import export_detector

    det = RollingMeanDetector()
    det.fit(_tiny_dataset(), FitMode.ZEROSHOT, {})
    with pytest.raises(NotImplementedError, match="register_onnx_wrapper"):
        export_detector(det, output_dir=Path("/tmp"), threshold=1.0, n_channels=3, window_length=64)
