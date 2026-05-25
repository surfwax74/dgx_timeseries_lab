"""Verify all bundled implementations self-register on import."""

from __future__ import annotations

import dgx_ts_lab  # noqa: F401  triggers self-registration
from dgx_ts_core.registry import DATASET_REGISTRY, DETECTOR_REGISTRY, TRAINER_REGISTRY


def test_trivial_synth_dataset_registered() -> None:
    assert "trivial_synth" in DATASET_REGISTRY.list()


def test_rolling_mean_detector_registered() -> None:
    assert "rolling_mean" in DETECTOR_REGISTRY.list()


def test_lightning_trainer_registered() -> None:
    assert "lightning" in TRAINER_REGISTRY.list()
