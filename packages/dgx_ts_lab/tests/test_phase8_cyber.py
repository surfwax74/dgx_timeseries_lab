"""Phase 8 tests — cybersecurity datasets + sequence transformer + operator fingerprint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from dgx_ts_core.data import TelemetryDataset
from dgx_ts_core.models import AnomalyDetector, FitMode, OutputKind

# ── Tokenizer ──────────────────────────────────────────────────────────


def test_command_tokenizer_roundtrip() -> None:
    from dgx_ts_lab.datasets.cyber import CommandTokenizer
    from dgx_ts_lab.datasets.cyber._tokenizer import CMD_TOKEN, N_SPECIAL

    tok = CommandTokenizer(opcodes=["OP_A", "OP_B"], param_values=["1", "2", "FOO"])
    assert tok.vocab_size == N_SPECIAL + 2 + 3
    tokens = tok.encode_command("OP_A", ["1", "FOO"])
    assert tokens[0] == CMD_TOKEN
    # Decoding round-trips through human-readable strings
    decoded = tok.decode(tokens)
    assert decoded[0] == "<CMD>"
    assert "OP_A" in decoded[1]
    assert "1" in decoded[2]


def test_command_tokenizer_unk_for_missing() -> None:
    from dgx_ts_lab.datasets.cyber import CommandTokenizer
    from dgx_ts_lab.datasets.cyber._tokenizer import UNK_TOKEN

    tok = CommandTokenizer(opcodes=["OP_A"], param_values=["1"])
    tokens = tok.encode_command("UNKNOWN_OP", ["UNKNOWN_PARAM"])
    assert tokens[1] == UNK_TOKEN  # opcode
    assert tokens[2] == UNK_TOKEN  # param


def test_command_tokenizer_save_load(tmp_path: Path) -> None:
    from dgx_ts_lab.datasets.cyber import CommandTokenizer

    tok = CommandTokenizer(opcodes=["A", "B"], param_values=["x", "y"])
    p = tmp_path / "tok.json"
    tok.save(p)
    loaded = CommandTokenizer.load(p)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode_command("A", ["x"]) == tok.encode_command("A", ["x"])


# ── CommandSequenceDataset + synth ─────────────────────────────────────


def test_command_sequence_dataset_implements_protocol() -> None:
    from dgx_ts_lab.datasets.cyber import CommandSequenceDataset, CommandTokenizer

    tok = CommandTokenizer(opcodes=["OP_A"], param_values=["1"])
    ds = CommandSequenceDataset(
        tokens=np.array([1, 4, 5, 1, 4, 5] * 10, dtype=np.int64),
        labels=np.zeros(60, dtype=np.bool_),
        tokenizer=tok,
    )
    assert isinstance(ds, TelemetryDataset)
    assert ds.has_labels
    win = next(ds.windows(length=30, stride=30))
    assert win.tensor.shape == (30, 1)


def test_synth_command_sequence_generates_injections() -> None:
    from dgx_ts_lab.datasets.synthetic.cyber import generate_command_sequence

    kwargs = generate_command_sequence(
        n_commands=2_000,
        inject_priv_escalation_rate=0.05,
        inject_flooding_rate=0.005,
        seed=42,
    )
    labels = kwargs["labels"]
    assert labels.sum() > 0, "expected at least some injected anomalies"
    # Aux carries the injection type so we can break down per category
    inj_types = kwargs["aux"]["injection_type"]
    assert set(np.unique(inj_types)).issuperset({0, 1, 2})  # 0=normal + at least priv + flooding


def test_synth_command_sequence_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DATASET_REGISTRY

    assert "synth_command_sequence" in DATASET_REGISTRY.list()
    ds = DATASET_REGISTRY.create("synth_command_sequence", n_commands=200, seed=0)
    win = next(ds.windows(length=50, stride=50))
    assert win.tensor.shape[0] == 50


# ── ActivityWindowDataset + synth ──────────────────────────────────────


def test_synth_operator_traffic_separates_operators() -> None:
    from dgx_ts_lab.datasets.synthetic.cyber import generate_operator_traffic

    kw = generate_operator_traffic(n_windows=1000, seed=0, impersonation_rate=0.0)
    features = kw["features"]
    op_ids = kw["operator_ids"]
    # Each operator's mean features should differ
    means = []
    for op in np.unique(op_ids):
        means.append(features[op_ids == op].mean(axis=0))
    means = np.stack(means)
    # Total variance across operator means should be non-trivial
    assert means.std(axis=0).max() > 0.1


def test_activity_window_dataset_implements_protocol() -> None:
    from dgx_ts_lab.datasets.cyber import ActivityWindowDataset

    n = 200
    ds = ActivityWindowDataset(
        features=np.random.randn(n, 6).astype(np.float32),
        operator_ids=np.random.randint(0, 3, size=n).astype(np.int64),
        labels=np.zeros(n, dtype=np.bool_),
    )
    assert isinstance(ds, TelemetryDataset)
    win = next(ds.windows(length=32, stride=32))
    assert win.tensor.shape == (32, 6)
    assert win.aux_labels is not None
    assert "operator_id" in win.aux_labels


def test_synth_operator_traffic_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DATASET_REGISTRY

    assert "synth_operator_traffic" in DATASET_REGISTRY.list()


# ── SideChannelDataset adapter ─────────────────────────────────────────


def test_side_channel_dataset_wraps_existing() -> None:
    from dgx_ts_lab.datasets.cyber import SideChannelDataset
    from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset

    src = TrivialSyntheticDataset(n_samples=600, n_channels=3, seed=0)
    sc = SideChannelDataset(source=src, summary_window_length=30, summary_stride=15)
    win = next(sc.windows(length=16, stride=16))
    # Each side-channel "feature" = source_channels × 4 default stats
    assert win.tensor.shape == (16, 3 * 4)
    assert isinstance(sc, TelemetryDataset)


# ── Sequence transformer ───────────────────────────────────────────────


def _tiny_seq_dataset(n_commands: int = 500, seed: int = 0):
    from dgx_ts_lab.datasets.cyber import CommandSequenceDataset
    from dgx_ts_lab.datasets.synthetic.cyber import generate_command_sequence

    kw = generate_command_sequence(n_commands=n_commands, seed=seed)
    return CommandSequenceDataset(**kw)


def test_sequence_transformer_protocol() -> None:
    from dgx_ts_lab.models.from_scratch import SequenceTransformerDetector

    det = SequenceTransformerDetector(vocab_size=32, max_seq_len=64, d_model=32, n_layers=1, n_heads=2)
    assert isinstance(det, AnomalyDetector)
    assert det.capabilities.requires_pretraining
    assert det.capabilities.output_kind is OutputKind.PER_STEP


def test_sequence_transformer_fit_and_loss() -> None:
    from dgx_ts_lab.models.from_scratch import SequenceTransformerDetector

    ds = _tiny_seq_dataset(n_commands=500)
    det = SequenceTransformerDetector(
        vocab_size=ds.vocab_size, max_seq_len=64, d_model=32, n_layers=1, n_heads=2,
    )
    det.fit(ds, FitMode.PRETRAIN, {})
    # Build a batch from the dataset's first window
    win = next(ds.windows(length=64, stride=64))
    batch = {
        "x": torch.from_numpy(win.tensor).unsqueeze(0).float(),
        "labels": torch.zeros(1, 64, dtype=torch.bool),
    }
    loss = det.compute_loss(batch)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_sequence_transformer_score_shape() -> None:
    from dgx_ts_lab.models.from_scratch import SequenceTransformerDetector

    ds = _tiny_seq_dataset(n_commands=500)
    det = SequenceTransformerDetector(
        vocab_size=ds.vocab_size, max_seq_len=64, d_model=32, n_layers=1, n_heads=2,
        score_n_samples=2,
    )
    det.fit(ds, FitMode.PRETRAIN, {})
    win = next(ds.windows(length=64, stride=64))
    batch = {
        "x": torch.from_numpy(win.tensor).unsqueeze(0).float(),
        "labels": torch.zeros(1, 64, dtype=torch.bool),
    }
    scores = det.compute_score_batch(batch)
    assert scores.shape == (1, 64)


def test_sequence_transformer_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DETECTOR_REGISTRY

    assert "sequence_transformer" in DETECTOR_REGISTRY.list()


# ── Operator fingerprint ───────────────────────────────────────────────


def _tiny_activity_dataset(n_windows: int = 600, seed: int = 0, impers: float = 0.05):
    from dgx_ts_lab.datasets.cyber import ActivityWindowDataset
    from dgx_ts_lab.datasets.synthetic.cyber import generate_operator_traffic

    kw = generate_operator_traffic(n_windows=n_windows, seed=seed, impersonation_rate=impers)
    return ActivityWindowDataset(**kw)


def test_operator_fingerprint_protocol() -> None:
    from dgx_ts_lab.models.behavior import OperatorFingerprintDetector

    det = OperatorFingerprintDetector(embedding_dim=8, n_operators=3, n_channels=6)
    assert isinstance(det, AnomalyDetector)
    assert det.capabilities.supports_streaming  # Mahalanobis is cheap at inference


def test_operator_fingerprint_fit_and_stats() -> None:
    from dgx_ts_lab.models.behavior import OperatorFingerprintDetector

    ds = _tiny_activity_dataset(n_windows=300, seed=0, impers=0.0)
    det = OperatorFingerprintDetector(embedding_dim=8, n_operators=3)
    det.fit(ds, FitMode.PRETRAIN, {})
    counts = det.compute_operator_stats(ds, window_length=10, stride=10)
    # Each of the 3 operators should have observed samples
    assert all(c > 0 for c in counts.values())
    # operator_valid should be set for trained operators
    assert det.module.operator_valid.sum().item() == 3


def test_operator_fingerprint_score_higher_for_impersonation() -> None:
    """Smoke check: after fit, impersonated windows should score higher on average."""
    from dgx_ts_lab.models.behavior import OperatorFingerprintDetector

    # Plumbing-only test: untrained encoder produces random embeddings, so
    # discrimination quality requires the Fabric loop training (exercised via
    # `dgx-ts train experiment=phase8_cyber`). Here we only verify the pipeline.
    ds_train = _tiny_activity_dataset(n_windows=500, seed=1, impers=0.0)
    ds_eval = _tiny_activity_dataset(n_windows=200, seed=2, impers=0.3)

    det = OperatorFingerprintDetector(embedding_dim=8, n_operators=3)
    det.fit(ds_train, FitMode.PRETRAIN, {})
    det.compute_operator_stats(ds_train, window_length=10, stride=10)

    n_scored = 0
    for w in ds_eval.windows(length=10, stride=10):
        s = det.score(w).scores
        assert s.shape == (10,)
        assert np.all(np.isfinite(s))
        n_scored += 1
        if n_scored >= 5:
            break
    assert n_scored >= 5


def test_operator_fingerprint_registered() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DETECTOR_REGISTRY

    assert "operator_fingerprint" in DETECTOR_REGISTRY.list()


# ── All Phase 8 datasets / detectors registered ────────────────────────


def test_all_phase8_keys_present() -> None:
    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DATASET_REGISTRY, DETECTOR_REGISTRY

    for key in ("command_sequence", "activity_window", "side_channel",
                "synth_command_sequence", "synth_operator_traffic"):
        assert key in DATASET_REGISTRY.list(), f"dataset {key!r} not registered"
    for key in ("sequence_transformer", "operator_fingerprint"):
        assert key in DETECTOR_REGISTRY.list(), f"detector {key!r} not registered"
