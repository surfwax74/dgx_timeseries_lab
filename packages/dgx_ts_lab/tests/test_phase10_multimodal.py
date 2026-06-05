"""Phase 10 tests — multi-modal (telemetry + commands + logs) foundation model.

Covers:
    - Log tokenizer + LogSeverity ordering
    - CommandEventBucketer / LogEventBucketer reduce sparse events to fixed-Hz frames
    - MultiModalDataset shape / split / window / channel_modalities provenance
    - synth_multimodal_leo registers + generates aligned modalities
    - Cross-modal blocks (ModalityTypeEmbedding, PerModalitySelfAttn, SharedCrossModalStack)
    - SatMultiModalDetector fit / compute_loss / compute_score_batch / save+load
    - DETECTOR_REGISTRY contains "sat_multimodal"
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dgx_ts_core.models import FitMode


# ── Log tokenizer ──────────────────────────────────────────────────────


def test_log_severity_ordering() -> None:
    from dgx_ts_lab.datasets.multimodal._log_tokenizer import LogSeverity

    assert int(LogSeverity.TRACE) < int(LogSeverity.INFO)
    assert int(LogSeverity.INFO) < int(LogSeverity.ERROR)
    assert int(LogSeverity.ERROR) < int(LogSeverity.FATAL)


def test_log_tokenizer_round_trip() -> None:
    from dgx_ts_lab.datasets.multimodal._log_tokenizer import (
        LOG_N_SPECIAL,
        LOG_UNK,
        LogTokenizer,
    )

    tok = LogTokenizer(sources=["OBDH", "EPS"], codes=["HEARTBEAT", "FAULT"])
    assert tok.source_id("OBDH") == LOG_N_SPECIAL + 0
    assert tok.source_id("EPS") == LOG_N_SPECIAL + 1
    assert tok.source_id("MYSTERY") == LOG_UNK
    assert tok.code_id("HEARTBEAT") == LOG_N_SPECIAL + len(tok.sources) + 0
    assert tok.vocab_size == LOG_N_SPECIAL + 2 + 2


# ── Event bucketers ────────────────────────────────────────────────────


def test_command_bucketer_shape_and_count() -> None:
    from dgx_ts_lab.datasets.multimodal.event_bucketer import CommandEventBucketer

    b = CommandEventBucketer(sample_rate_hz=1.0)
    times = np.array([0.1, 0.4, 5.5], dtype=np.float64)
    op = np.array([1, 2, 3], dtype=np.int64)
    pm = np.array([4, 5, 6], dtype=np.int64)
    out = b.bucket(times, op, pm, n_bins=10)
    assert out.shape == (10, 3)
    assert out.dtype == np.float32
    # Bin 0 saw 2 events; last opcode = 2, last param = 5
    assert out[0, 0] == 2.0
    assert out[0, 1] == 2.0
    assert out[0, 2] == 5.0
    # Bin 5 saw 1 event
    assert out[5, 0] == 1.0
    assert out[5, 1] == 3.0
    # Bin 7 saw none
    assert out[7].sum() == 0.0


def test_command_bucketer_empty_safe() -> None:
    from dgx_ts_lab.datasets.multimodal.event_bucketer import CommandEventBucketer

    b = CommandEventBucketer(sample_rate_hz=1.0)
    out = b.bucket(
        np.zeros(0, dtype=np.float64),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        n_bins=5,
    )
    assert out.shape == (5, 3)
    assert out.sum() == 0.0


def test_log_bucketer_max_severity() -> None:
    from dgx_ts_lab.datasets.multimodal._log_tokenizer import LogSeverity
    from dgx_ts_lab.datasets.multimodal.event_bucketer import LogEventBucketer

    b = LogEventBucketer(sample_rate_hz=1.0)
    times = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    sev = np.array(
        [int(LogSeverity.INFO), int(LogSeverity.FATAL), int(LogSeverity.WARN)],
        dtype=np.int64,
    )
    src = np.array([1, 2, 3], dtype=np.int64)
    out = b.bucket(times, sev, src, n_bins=2)
    # Bin 0 should record FATAL as max severity
    assert out[0, 0] == 3.0
    assert out[0, 1] == float(int(LogSeverity.FATAL))


# ── MultiModalDataset ─────────────────────────────────────────────────


def _build_multimodal(n_seconds: int = 200) -> "MultiModalDataset":  # type: ignore[name-defined]
    from dgx_ts_lab.datasets.multimodal.synth_multimodal_leo import (
        generate_multimodal_leo,
    )
    from dgx_ts_lab.datasets.multimodal.multimodal_dataset import MultiModalDataset

    kwargs = generate_multimodal_leo(n_seconds=n_seconds, seed=0, n_telemetry_channels=4)
    return MultiModalDataset(**kwargs)


def test_multimodal_dataset_channel_layout() -> None:
    ds = _build_multimodal(n_seconds=200)
    # 4 telemetry + 3 commands + 3 logs = 10
    assert len(ds.channels) == 10
    assert ds.n_telemetry_channels == 4
    assert ds.channel_modalities == (
        "telemetry", "telemetry", "telemetry", "telemetry",
        "command", "command", "command",
        "log", "log", "log",
    )


def test_multimodal_dataset_window_carries_modality_provenance() -> None:
    ds = _build_multimodal(n_seconds=200)
    win = next(ds.windows(length=64, stride=64))
    assert win.tensor.shape == (64, 10)
    assert win.provenance["channel_modalities"] == ds.channel_modalities
    assert win.provenance["n_telemetry_channels"] == 4


def test_multimodal_dataset_split_preserves_modalities() -> None:
    from dgx_ts_core.data import SplitScheme, SplitStrategy

    ds = _build_multimodal(n_seconds=400)
    parts = ds.split(
        SplitScheme(strategy=SplitStrategy.TEMPORAL, train_frac=0.6, val_frac=0.2, test_frac=0.2)
    )
    for name in ("train", "val", "test"):
        assert parts[name].channel_modalities == ds.channel_modalities
        assert parts[name].n_telemetry_channels == 4


def test_multimodal_dataset_split_window_by_modality() -> None:
    ds = _build_multimodal(n_seconds=200)
    win = next(ds.windows(length=64, stride=64))
    parts = ds.split_window_by_modality(win)
    assert parts["telemetry"].shape == (64, 4)
    assert parts["commands"].shape == (64, 3)
    assert parts["logs"].shape == (64, 3)


# ── synth registry ─────────────────────────────────────────────────────


def test_synth_multimodal_leo_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DATASET_REGISTRY

    assert "synth_multimodal_leo" in DATASET_REGISTRY.list()


def test_synth_multimodal_leo_create_via_registry() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DATASET_REGISTRY

    ds = DATASET_REGISTRY.create(
        "synth_multimodal_leo",
        n_seconds=200,
        seed=0,
        n_telemetry_channels=3,
    )
    assert ds.n_telemetry_channels == 3
    assert len(ds.channels) == 3 + 3 + 3


# ── Cross-modal building blocks ────────────────────────────────────────


def test_modality_type_embedding_adds_per_token_bias() -> None:
    from dgx_ts_lab.models.from_scratch._multimodal_blocks import (
        ModalityTypeEmbedding,
    )

    me = ModalityTypeEmbedding(n_modalities=3, d_model=16)
    x = torch.zeros(2, 6, 16)
    ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    out = me(x, ids)
    assert out.shape == (2, 6, 16)
    # Different modality ids produce different biases
    assert not torch.allclose(out[0, 0], out[0, 2])
    assert not torch.allclose(out[0, 2], out[0, 4])


def test_per_modality_self_attn_runs() -> None:
    from dgx_ts_lab.models.from_scratch._multimodal_blocks import (
        PerModalitySelfAttn,
    )

    enc = PerModalitySelfAttn(d_model=16, n_heads=2, d_ff=32, n_layers=1)
    x = torch.randn(2, 4, 16)
    out = enc(x)
    assert out.shape == x.shape


def test_shared_cross_modal_stack_runs() -> None:
    from dgx_ts_lab.models.from_scratch._multimodal_blocks import (
        SharedCrossModalStack,
    )

    enc = SharedCrossModalStack(d_model=16, n_heads=2, d_ff=32, n_layers=2)
    x = torch.randn(2, 12, 16)
    out = enc(x)
    assert out.shape == x.shape


# ── SatMultiModalDetector ──────────────────────────────────────────────


def _tiny_multimodal_detector():
    from dgx_ts_lab.models.from_scratch import SatMultiModalDetector

    return SatMultiModalDetector(
        window_length=64,
        patch_len=16,
        d_model=16,
        n_heads=2,
        n_layers_self_attn_per_modality=1,
        n_layers_cross_modal=1,
        d_ff=32,
    )


def test_sat_multimodal_detector_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DETECTOR_REGISTRY

    assert "sat_multimodal" in DETECTOR_REGISTRY.list()


def test_sat_multimodal_capabilities_declare_pretraining() -> None:
    det = _tiny_multimodal_detector()
    caps = det.capabilities
    assert caps.requires_pretraining is True
    assert caps.supports_multivariate is True
    assert caps.native_context_len == 64


def test_sat_multimodal_fit_builds_module_and_populates_norm() -> None:
    det = _tiny_multimodal_detector()
    ds = _build_multimodal(n_seconds=200)
    res = det.fit(ds, FitMode.PRETRAIN, {})
    assert det.module is not None
    assert res.metadata["n_telemetry_channels"] == 4
    assert res.metadata["command_features"] == 3
    assert res.metadata["log_features"] == 3
    # tel_std should be > 0 after copying dataset stats
    assert (det.module.tel_std > 0).all()


def test_sat_multimodal_compute_loss_runs() -> None:
    det = _tiny_multimodal_detector()
    ds = _build_multimodal(n_seconds=200)
    det.fit(ds, FitMode.PRETRAIN, {})
    # Build a batch from the dataset: tensor has 4+3+3 = 10 channels
    win = next(ds.windows(length=64, stride=64))
    batch = {"x": torch.from_numpy(win.tensor).float().unsqueeze(0)}
    loss = det.compute_loss(batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_sat_multimodal_compute_score_batch_shape() -> None:
    det = _tiny_multimodal_detector()
    ds = _build_multimodal(n_seconds=200)
    det.fit(ds, FitMode.PRETRAIN, {})
    win = next(ds.windows(length=64, stride=64))
    batch = {"x": torch.from_numpy(win.tensor).float().unsqueeze(0)}
    scores = det.compute_score_batch(batch)
    assert scores.shape == (1, 64)
    assert torch.isfinite(scores).all()


def test_sat_multimodal_score_single_window() -> None:
    det = _tiny_multimodal_detector()
    ds = _build_multimodal(n_seconds=200)
    det.fit(ds, FitMode.PRETRAIN, {})
    win = next(ds.windows(length=64, stride=64))
    s = det.score(win)
    assert s.scores.shape == (64,)
    assert np.isfinite(s.scores).all()


def test_sat_multimodal_save_load(tmp_path) -> None:
    from dgx_ts_lab.models.from_scratch import SatMultiModalDetector

    det = _tiny_multimodal_detector()
    ds = _build_multimodal(n_seconds=200)
    det.fit(ds, FitMode.PRETRAIN, {})
    p = tmp_path / "sm.pt"
    det.save(p)
    loaded = SatMultiModalDetector.load(p)
    assert loaded.module is not None
    # Same-input scores should match (eval mode, deterministic)
    win = next(ds.windows(length=64, stride=64))
    batch = {"x": torch.from_numpy(win.tensor).float().unsqueeze(0)}
    a = det.compute_score_batch(batch)
    b = loaded.compute_score_batch(batch)
    torch.testing.assert_close(a, b)


def test_sat_multimodal_window_length_must_be_multiple_of_patch_len() -> None:
    from dgx_ts_lab.models.from_scratch import SatMultiModalDetector

    det = SatMultiModalDetector(
        window_length=70,   # not divisible by patch_len 16
        patch_len=16,
        d_model=16,
        n_heads=2,
    )
    ds = _build_multimodal(n_seconds=200)
    with pytest.raises(ValueError, match="multiple of patch_len"):
        det.fit(ds, FitMode.PRETRAIN, {})
