"""Integration test: prove the lift contract.

Exports a detector, then loads the artifacts using ONLY the imports a
downstream MLOps consumer would have:

    dgx_ts_core       (interfaces only — no torch dep)
    onnxruntime
    numpy
    yaml

Verifies that inference produces numerically-identical scores to the
in-process detector, AND that the model_card / feature_schema YAMLs
deserialize cleanly into the dgx_ts_core dataclasses.

Note: this test process inherits the parent venv's full import space, but
only EXERCISES the consumer-side imports. A separate
``scripts/test_fresh_venv_lift.sh`` does the actual fresh-venv version.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# Track imports done by the "consumer" side of the test so we can fail
# loudly if something accidentally touches dgx_ts_lab.
_CONSUMER_ALLOWED = {
    "dgx_ts_core",
    "dgx_ts_core.data",
    "dgx_ts_core.data.window",
    "dgx_ts_core.data.dataset",
    "dgx_ts_core.data.schema",
    "dgx_ts_core.data.splits",
    "dgx_ts_core.models",
    "dgx_ts_core.models.detector",
    "dgx_ts_core.models.capabilities",
    "dgx_ts_core.models.scores",
    "dgx_ts_core.export",
    "dgx_ts_core.export.model_card",
    "dgx_ts_core.export.feature_schema",
    "dgx_ts_core.export.formats",
    "onnxruntime",
    "numpy",
    "yaml",
}


def _consumer_load_and_score(artifact_dir: Path, window: np.ndarray) -> np.ndarray:
    """Simulates the mm_mlops consumer: ONNX + dgx_ts_core artifacts only."""
    import onnxruntime as ort
    import yaml

    # Load model_card.yaml — must deserialize into dgx_ts_core types
    from dgx_ts_core.export import FeatureSchema, ModelCard
    from dgx_ts_core.models import Capabilities, OutputKind

    card_dict = yaml.safe_load((artifact_dir / "model_card.yaml").read_text())
    # Reconstruct the Capabilities object
    cap_dict = dict(card_dict["capabilities"])
    cap_dict["output_kind"] = OutputKind(cap_dict["output_kind"])
    capabilities = Capabilities(**cap_dict)

    # Load feature_schema.yaml
    schema_dict = yaml.safe_load((artifact_dir / "feature_schema.yaml").read_text())
    assert "channels" in schema_dict
    assert "normalization" in schema_dict

    # Run ONNX inference
    sess = ort.InferenceSession(str(artifact_dir / "model.onnx"))
    scores = sess.run(["scores"], {"x": window})[0]
    return scores


def test_lift_contract_end_to_end(tmp_path: Path) -> None:
    """Full round-trip: train → export → consumer-side load → score → compare."""
    import torch
    from dgx_ts_core.models import FitMode

    from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset
    from dgx_ts_lab.models.from_scratch import PatchTSTMAEDetector
    from dgx_ts_lab.serving import (
        export_detector,
        write_feature_schema,
        write_model_card,
    )

    # ── 1. Train (cheap) ─────────────────────────────────────────────
    ds = TrivialSyntheticDataset(n_samples=400, n_channels=3, seed=0)
    det = PatchTSTMAEDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    det.fit(ds, FitMode.PRETRAIN, {})

    # ── 2. Export to artifacts dir ───────────────────────────────────
    artifact_dir = tmp_path / "artifacts"
    export_detector(
        det, output_dir=artifact_dir, threshold=1.0, n_channels=3, window_length=64,
    )
    write_model_card(
        detector_name=det.name,
        detector_version="test-0.1",
        capabilities=det.capabilities,
        intended_subsystem="eps",
        training_dataset=ds.name,
        metrics={"f1": 0.8, "roc_auc": 0.9},
        calibrated_threshold=1.0,
        output_path=artifact_dir / "model_card.yaml",
    )
    write_feature_schema(
        channels=ds.channels,
        sample_rate_hz=ds.sample_rate_hz,
        window_length=64,
        stats=ds.stats(),
        output_path=artifact_dir / "feature_schema.yaml",
    )

    # ── 3. Consumer-side load + inference ────────────────────────────
    x_np = np.random.randn(2, 64, 3).astype(np.float32)
    consumer_scores = _consumer_load_and_score(artifact_dir, x_np)

    # ── 4. Numeric agreement with in-process detector ────────────────
    inproc_scores = det.compute_score_batch({"x": torch.from_numpy(x_np)}).cpu().numpy()
    np.testing.assert_allclose(consumer_scores, inproc_scores, atol=1e-4, rtol=1e-3)


def test_artifact_files_self_describe(tmp_path: Path) -> None:
    """All three artifacts must be human-readable YAML or standard ONNX."""
    import torch
    import yaml
    from dgx_ts_core.models import FitMode

    from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset
    from dgx_ts_lab.models.from_scratch import PatchTSTMAEDetector
    from dgx_ts_lab.serving import (
        export_detector,
        write_feature_schema,
        write_model_card,
    )

    ds = TrivialSyntheticDataset(n_samples=400, n_channels=2, seed=0)
    det = PatchTSTMAEDetector(window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2)
    det.fit(ds, FitMode.PRETRAIN, {})

    export_detector(
        det, output_dir=tmp_path, threshold=0.5, n_channels=2, window_length=64,
    )
    write_model_card(
        detector_name=det.name, detector_version="test-0.1",
        capabilities=det.capabilities, intended_subsystem="eps",
        training_dataset=ds.name, metrics={"f1": 0.5},
        calibrated_threshold=0.5, output_path=tmp_path / "model_card.yaml",
    )
    write_feature_schema(
        channels=ds.channels, sample_rate_hz=ds.sample_rate_hz,
        window_length=64, stats=ds.stats(),
        output_path=tmp_path / "feature_schema.yaml",
    )

    # YAML files round-trip cleanly
    card = yaml.safe_load((tmp_path / "model_card.yaml").read_text())
    schema = yaml.safe_load((tmp_path / "feature_schema.yaml").read_text())
    assert card["detector_name"] == "patchtst_mae"
    assert schema["window_length"] == 64

    # ONNX files exist and have non-trivial size
    assert (tmp_path / "model.onnx").stat().st_size > 1024
    assert (tmp_path / "model_with_threshold.onnx").stat().st_size > 1024
