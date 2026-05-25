"""Phase 2 neural detector tests.

Covers: protocol conformance, Capabilities honesty, fit → module exists,
compute_loss returns scalar, compute_score_batch returns (B, T), score on
TelemetryWindow round-trips, save → load identity check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dgx_ts_core.models import AnomalyDetector, FitMode, OutputKind
from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset
from dgx_ts_lab.models.from_scratch import (
    AnomalyTransformerDetector,
    DCdetectorDetector,
    PatchTSTMAEDetector,
)

WINDOW = 64
PATCH = 16


def _tiny_dataset(n_channels: int = 3) -> TrivialSyntheticDataset:
    return TrivialSyntheticDataset(
        n_samples=400, n_channels=n_channels, anomaly_rate=0.02,
        anomaly_magnitude_sigmas=10.0, seed=7,
    )


def _tiny_batch(n_channels: int = 3, batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(batch_size, WINDOW, n_channels, dtype=torch.float32),
        "labels": torch.zeros(batch_size, WINDOW, dtype=torch.bool),
    }


# ── PatchTST+MAE ────────────────────────────────────────────────────────


def test_patchtst_implements_protocol() -> None:
    det = PatchTSTMAEDetector(window_length=WINDOW, patch_len=PATCH, d_model=32, n_layers=1, n_heads=2)
    assert isinstance(det, AnomalyDetector)
    caps = det.capabilities
    assert caps.requires_pretraining is True
    assert caps.output_kind is OutputKind.PER_STEP
    assert caps.native_context_len == WINDOW


def test_patchtst_fit_builds_module_with_correct_channels() -> None:
    det = PatchTSTMAEDetector(window_length=WINDOW, patch_len=PATCH, d_model=32, n_layers=1, n_heads=2)
    ds = _tiny_dataset(n_channels=3)
    det.fit(ds, FitMode.PRETRAIN, {})
    assert det.module is not None
    assert det.module.n_channels == 3
    # Normalization buffers populated
    assert not torch.equal(det.module.norm_std, torch.ones(3))


def test_patchtst_compute_loss_returns_scalar() -> None:
    det = PatchTSTMAEDetector(window_length=WINDOW, patch_len=PATCH, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    loss = det.compute_loss(_tiny_batch())
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_patchtst_compute_score_batch_shape() -> None:
    det = PatchTSTMAEDetector(window_length=WINDOW, patch_len=PATCH, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    batch = _tiny_batch(batch_size=5)
    scores = det.compute_score_batch(batch)
    assert scores.shape == (5, WINDOW)


def test_patchtst_mask_ratio_varies_across_batches() -> None:
    """Per-batch ratio must be randomized, otherwise the encoder never sees
    low-/no-mask inputs and eval suffers a train/eval distribution shift
    (the bug that exploded val_loss on the LEO EPS preset)."""
    det = PatchTSTMAEDetector(
        window_length=WINDOW, patch_len=PATCH, d_model=32, n_layers=1,
        n_heads=2, mask_ratio=0.8,
    )
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    assert det.module is not None
    torch.manual_seed(0)
    densities: set[int] = set()
    for _ in range(20):
        x = torch.randn(2, WINDOW, 3, dtype=torch.float32)
        _, mask = det.module(x, apply_mask=True)
        assert mask is not None
        densities.add(int(mask.float().mean().item() * 100))
    # 20 draws from U(0, 0.8) almost surely span >=3 distinct density buckets.
    assert len(densities) >= 3, f"mask ratio looks constant: {densities}"


def test_patchtst_compute_loss_finite_when_ratio_collapses_to_zero() -> None:
    """With mask_ratio=0 the sampled ratio is always 0, mask is all-False —
    loss must still be finite (falls back to all-patches MSE) so gradient flows."""
    det = PatchTSTMAEDetector(
        window_length=WINDOW, patch_len=PATCH, d_model=32, n_layers=1,
        n_heads=2, mask_ratio=0.0,
    )
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    loss = det.compute_loss(_tiny_batch())
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0  # untrained model — reconstruction is non-trivial


def test_patchtst_save_load_identity(tmp_path: Path) -> None:
    det = PatchTSTMAEDetector(window_length=WINDOW, patch_len=PATCH, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    p = tmp_path / "patchtst.pt"
    det.save(p)
    loaded = PatchTSTMAEDetector.load(p)
    batch = _tiny_batch()
    a = det.compute_score_batch(batch)
    b = loaded.compute_score_batch(batch)
    torch.testing.assert_close(a, b)


# ── Anomaly Transformer ─────────────────────────────────────────────────


def test_anomaly_transformer_implements_protocol() -> None:
    det = AnomalyTransformerDetector(window_length=WINDOW, d_model=32, n_layers=1, n_heads=2)
    assert isinstance(det, AnomalyDetector)
    assert det.capabilities.requires_pretraining is True


def test_anomaly_transformer_compute_loss() -> None:
    det = AnomalyTransformerDetector(window_length=WINDOW, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    loss = det.compute_loss(_tiny_batch())
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_anomaly_transformer_score_shape() -> None:
    det = AnomalyTransformerDetector(window_length=WINDOW, d_model=32, n_layers=1, n_heads=2)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    scores = det.compute_score_batch(_tiny_batch(batch_size=3))
    assert scores.shape == (3, WINDOW)


# ── DCdetector ──────────────────────────────────────────────────────────


def test_dcdetector_implements_protocol() -> None:
    det = DCdetectorDetector(window_length=WINDOW, patch_len=PATCH, d_model=16)
    assert isinstance(det, AnomalyDetector)
    assert det.capabilities.requires_pretraining is True


def test_dcdetector_compute_loss() -> None:
    det = DCdetectorDetector(window_length=WINDOW, patch_len=PATCH, d_model=16)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    loss = det.compute_loss(_tiny_batch())
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_dcdetector_score_shape() -> None:
    det = DCdetectorDetector(window_length=WINDOW, patch_len=PATCH, d_model=16)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    scores = det.compute_score_batch(_tiny_batch(batch_size=2))
    assert scores.shape == (2, WINDOW)


# ── Registry wiring ─────────────────────────────────────────────────────


def test_all_three_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DETECTOR_REGISTRY

    keys = DETECTOR_REGISTRY.list()
    assert "patchtst_mae" in keys
    assert "anomaly_transformer" in keys
    assert "dcdetector" in keys
