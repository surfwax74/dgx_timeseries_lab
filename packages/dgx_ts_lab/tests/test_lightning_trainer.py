"""Unit tests for LightningTrainer Phase 0 path."""

from __future__ import annotations

from pathlib import Path

from dgx_ts_core.models import FitMode
from dgx_ts_core.training import TrainConfig
from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset
from dgx_ts_lab.models.baseline import RollingMeanDetector
from dgx_ts_lab.training import LightningTrainer


def test_classical_detector_fit_returns_val_test_metrics(tmp_path: Path) -> None:
    ds = TrivialSyntheticDataset(
        n_samples=3000, n_channels=3, anomaly_rate=0.02,
        anomaly_magnitude_sigmas=10.0, seed=11,
    )
    det = RollingMeanDetector()
    trainer = LightningTrainer()
    config = TrainConfig(
        window_length=128,
        window_stride=128,
        device="cpu",
        checkpoint_dir=tmp_path / "ckpt",
    )

    result = trainer.fit(det, ds, FitMode.ZEROSHOT, config)

    assert result.detector_name == "rolling_mean"
    assert "threshold" in result.metadata
    val = result.metadata["val_metrics"]
    test = result.metadata["test_metrics"]
    assert "f1" in val and "f1" in test
    # ROC-AUC is threshold-independent — checks whether the detector ranks
    # anomalies above non-anomalies. The smoke test uses a deliberately
    # small dataset (3k samples → 2.1k train), so AUC > 0.70 is the bar
    # for proving the trainer wires fit + scoring + eval through correctly.
    assert val["roc_auc"] > 0.70, f"val ROC-AUC {val['roc_auc']:.3f} below 0.70"
