"""Phase 3 tests — foundation-model adapter shells.

These tests don't require real foundation-model weights — the adapters
fall back to a small randomly-initialized T5 when weights are missing.
This makes the tests runnable on any machine including the air-gapped
DGX before weights are provisioned.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dgx_ts_core.models import AnomalyDetector, FitMode, OutputKind


def _tiny_dataset(n_channels: int = 3):
    from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset

    return TrivialSyntheticDataset(
        n_samples=400, n_channels=n_channels, anomaly_rate=0.02,
        anomaly_magnitude_sigmas=10.0, seed=0,
    )


def _tiny_batch(n_channels: int = 3, T: int = 64, B: int = 4):
    return {
        "x": torch.randn(B, T, n_channels, dtype=torch.float32),
        "labels": torch.zeros(B, T, dtype=torch.bool),
    }


# ── _loader ──────────────────────────────────────────────────────────────


def test_loader_resolves_existing_path(tmp_path: Path) -> None:
    from dgx_ts_lab.models.foundation._loader import resolve_model_path

    target = tmp_path / "fake_model"
    target.mkdir()
    # Direct absolute path → returned as-is
    assert resolve_model_path(str(target)) == target


def test_loader_raises_when_missing() -> None:
    from dgx_ts_lab.models.foundation._loader import resolve_model_path

    with pytest.raises(FileNotFoundError, match="foundation_model_provisioning"):
        resolve_model_path("nonexistent_org/nonexistent_model_xyz_unique_123")


# ── Registry wiring ──────────────────────────────────────────────────────


def test_all_three_foundation_models_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DETECTOR_REGISTRY

    keys = DETECTOR_REGISTRY.list()
    assert "chronos" in keys
    assert "moment" in keys
    assert "moirai" in keys


# ── Chronos ──────────────────────────────────────────────────────────────


def test_chronos_implements_protocol() -> None:
    from dgx_ts_lab.models.foundation import ChronosDetector

    det = ChronosDetector(window_length=64, n_quantile_bins=128)
    assert isinstance(det, AnomalyDetector)
    caps = det.capabilities
    assert caps.requires_pretraining is True
    assert caps.supports_peft is True
    assert caps.output_kind is OutputKind.PER_STEP


def test_chronos_fit_builds_module() -> None:
    from dgx_ts_lab.models.foundation import ChronosDetector

    det = ChronosDetector(window_length=64, n_quantile_bins=128)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    assert det.module is not None
    # Without real Chronos weights, falls back to untrained T5
    assert not det.module._is_pretrained


def test_chronos_compute_loss_returns_scalar() -> None:
    from dgx_ts_lab.models.foundation import ChronosDetector

    det = ChronosDetector(window_length=64, n_quantile_bins=128)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    loss = det.compute_loss(_tiny_batch())
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_chronos_score_shape() -> None:
    from dgx_ts_lab.models.foundation import ChronosDetector

    det = ChronosDetector(window_length=64, n_quantile_bins=128)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    scores = det.compute_score_batch(_tiny_batch(B=3))
    assert scores.shape == (3, 64)


# ── MOMENT ───────────────────────────────────────────────────────────────


def test_moment_implements_protocol() -> None:
    from dgx_ts_lab.models.foundation import MomentDetector

    det = MomentDetector(window_length=64, patch_len=8, d_model=32)
    assert isinstance(det, AnomalyDetector)
    assert det.capabilities.requires_pretraining is True


def test_moment_compute_loss() -> None:
    from dgx_ts_lab.models.foundation import MomentDetector

    det = MomentDetector(window_length=64, patch_len=8, d_model=32)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    loss = det.compute_loss(_tiny_batch())
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_moment_score_shape() -> None:
    from dgx_ts_lab.models.foundation import MomentDetector

    det = MomentDetector(window_length=64, patch_len=8, d_model=32)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    scores = det.compute_score_batch(_tiny_batch(B=2))
    assert scores.shape == (2, 64)


# ── Moirai (shell — requires uni2ts) ─────────────────────────────────────


def test_moirai_construction_works() -> None:
    """Moirai constructs without errors even without uni2ts; fit() is where it would fail."""
    from dgx_ts_lab.models.foundation import MoiraiDetector

    det = MoiraiDetector(window_length=64)
    assert isinstance(det, AnomalyDetector)
    # capabilities accessible without uni2ts
    assert det.capabilities.requires_pretraining is True


def test_moirai_fit_raises_clearly_without_uni2ts() -> None:
    """If uni2ts is not installed, fit() raises with a clear pointer."""
    from dgx_ts_lab.models.foundation import MoiraiDetector
    from dgx_ts_lab.models.foundation.moirai import _try_import_uni2ts

    if _try_import_uni2ts():
        pytest.skip("uni2ts is installed; the missing-import error path can't be exercised")
    det = MoiraiDetector(window_length=64)
    with pytest.raises(ImportError, match="uni2ts"):
        det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})


# ── PEFT helper ──────────────────────────────────────────────────────────


def test_lora_wrap_reduces_trainable_params() -> None:
    """LoRA wrapping should drastically reduce trainable param count."""
    from dgx_ts_lab.training.peft import (
        LoraConfig,
        trainable_parameter_count,
        wrap_with_lora,
    )
    from dgx_ts_lab.models.foundation import ChronosDetector

    det = ChronosDetector(window_length=64, n_quantile_bins=128)
    det.fit(_tiny_dataset(), FitMode.PRETRAIN, {})
    trainable_before, total_before = trainable_parameter_count(det.module)

    wrapped = wrap_with_lora(det.module, LoraConfig(r=4, alpha=8, target_modules=("q", "v")))
    trainable_after, total_after = trainable_parameter_count(wrapped)

    # LoRA freezes original params and adds small adapter params
    assert trainable_after < trainable_before
    # Trainable fraction should be very small (< 5%)
    assert trainable_after / total_after < 0.05
