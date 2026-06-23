"""Unit tests for RollingMeanDetector."""

from __future__ import annotations

import numpy as np
import pytest
from dgx_ts_core.models import AnomalyDetector, FitMode, OutputKind
from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset
from dgx_ts_lab.models.baseline import RollingMeanDetector


def test_detector_implements_protocol() -> None:
    det = RollingMeanDetector()
    assert isinstance(det, AnomalyDetector)
    caps = det.capabilities
    assert caps.requires_pretraining is False
    assert caps.supports_streaming is True
    assert caps.output_kind is OutputKind.PER_STEP


def test_score_before_fit_raises() -> None:
    det = RollingMeanDetector()
    ds = TrivialSyntheticDataset(n_samples=200, n_channels=2, seed=0)
    win = next(ds.windows(length=64, stride=64))
    with pytest.raises(RuntimeError, match="fit"):
        det.score(win)


def test_fit_score_round_trip_flags_injected_spikes() -> None:
    from sklearn.metrics import roc_auc_score

    ds = TrivialSyntheticDataset(
        n_samples=5000,
        n_channels=3,
        noise_std=0.05,
        anomaly_rate=0.02,
        anomaly_magnitude_sigmas=10.0,
        seed=7,
    )
    det = RollingMeanDetector()
    det.fit(ds, FitMode.ZEROSHOT, {})

    all_scores = []
    all_labels = []
    for w in ds.windows(length=256, stride=256):
        s = det.score(w)
        all_scores.append(s.scores)
        all_labels.append(w.labels)
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)  # type: ignore[arg-type]

    # ROC-AUC is threshold-free: it measures whether the score ordering
    # separates anomalies from non-anomalies. For a working detector on this
    # easy synthetic problem we expect well above chance (0.5).
    auc = roc_auc_score(labels, scores)
    assert auc > 0.80, f"ROC-AUC {auc:.3f} below 0.80 — detector is barely separating anomalies"


def test_save_load_round_trip(tmp_path) -> None:
    ds = TrivialSyntheticDataset(n_samples=500, n_channels=2, seed=1)
    det = RollingMeanDetector()
    det.fit(ds, FitMode.ZEROSHOT, {})

    p = tmp_path / "det.npz"
    det.save(p)
    loaded = RollingMeanDetector.load(p)

    win = next(ds.windows(length=64, stride=64))
    np.testing.assert_array_equal(det.score(win).scores, loaded.score(win).scores)
