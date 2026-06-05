"""Phase 7 tests — coupling graph, attribution, cascade walker, report writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dgx_ts_core.models import FitMode


# ── CouplingGraph: declared path ───────────────────────────────────────


def _layered_with_coupling():
    """Build a tiny LayeredSyntheticDataset with explicit coupling components."""
    from dgx_ts_core.data import Channel, Subsystem, Units
    from dgx_ts_lab.datasets.synthetic.layered import (
        LayeredSyntheticDataset,
        coupling,
        noise,
        physics,
    )

    channels = (
        Channel(name="src", units=Units.DIMENSIONLESS, subsystem=Subsystem.EPS, sample_rate_hz=1.0),
        Channel(name="tgt", units=Units.DIMENSIONLESS, subsystem=Subsystem.EPS, sample_rate_hz=1.0),
        Channel(name="other", units=Units.DIMENSIONLESS, subsystem=Subsystem.EPS, sample_rate_hz=1.0),
    )
    components = [
        physics.OrbitalSinusoid("src", amplitude=1.0, period_s=200.0),
        coupling.LinearCoupling(source="src", target="tgt", gain=0.7),
        noise.GaussianNoise("tgt", std=0.05),
    ]
    return LayeredSyntheticDataset(channels, components, n_samples=500, seed=0)


def test_coupling_graph_declared_extracts_linear_coupling() -> None:
    from dgx_ts_lab.explanation import build_coupling_graph

    ds = _layered_with_coupling()
    g = build_coupling_graph(ds, strategy="auto")
    assert g.source == "declared"
    assert g.n_edges() >= 1
    # tgt should have src as an upstream edge
    edges = g.upstream("tgt")
    assert any(e.source == "src" for e in edges)


def test_coupling_graph_learned_from_correlations() -> None:
    from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset
    from dgx_ts_lab.explanation import build_coupling_graph

    ds = TrivialSyntheticDataset(n_samples=2000, n_channels=3, seed=0)
    g = build_coupling_graph(ds, strategy="learned", correlation_threshold=0.1)
    # Even uncorrelated sinusoids will produce *some* correlation; mostly
    # we just want the path to not blow up + to return source="learned".
    assert g.source == "learned"


def test_coupling_graph_force_strategy_falls_through() -> None:
    """`strategy=learned` should bypass the declared path."""
    from dgx_ts_lab.explanation import build_coupling_graph

    ds = _layered_with_coupling()
    g = build_coupling_graph(ds, strategy="learned", correlation_threshold=0.05)
    assert g.source == "learned"


def test_coupling_graph_none_returns_empty() -> None:
    from dgx_ts_lab.explanation import build_coupling_graph

    ds = _layered_with_coupling()
    g = build_coupling_graph(ds, strategy="none")
    assert g.source == "none"
    assert g.n_edges() == 0


# ── Attribution ────────────────────────────────────────────────────────


def test_attribution_classical_detector_falls_back_to_permutation() -> None:
    from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset
    from dgx_ts_lab.explanation import attribute_window
    from dgx_ts_lab.models.baseline import RollingMeanDetector

    ds = TrivialSyntheticDataset(n_samples=400, n_channels=4, seed=0)
    det = RollingMeanDetector()
    det.fit(ds, FitMode.ZEROSHOT, {})
    win = next(ds.windows(length=64, stride=64))
    ranked = attribute_window(det, win, n_permutation_trials=2)
    assert len(ranked) == 4
    assert ranked[0].rank == 1
    # Top-1 should have score 1.0 by normalization (if any signal exists)
    assert ranked[0].score == 1.0 or all(r.score == 0.0 for r in ranked)


def test_attribution_neural_detector_uses_integrated_gradients() -> None:
    """For a differentiable detector, the IG path should engage (or fall back gracefully)."""
    from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset
    from dgx_ts_lab.explanation import attribute_window
    from dgx_ts_lab.models.from_scratch import PatchTSTMAEDetector

    ds = TrivialSyntheticDataset(n_samples=400, n_channels=3, seed=0)
    det = PatchTSTMAEDetector(
        window_length=64, patch_len=16, d_model=32, n_layers=1, n_heads=2,
    )
    det.fit(ds, FitMode.PRETRAIN, {})
    win = next(ds.windows(length=64, stride=64))
    ranked = attribute_window(det, win, n_steps_ig=4)
    assert len(ranked) == 3
    assert ranked[0].rank == 1


# ── Cascade walker ─────────────────────────────────────────────────────


def test_cascade_walker_traces_one_hop() -> None:
    from dgx_ts_lab.explanation import walk_cascade
    from dgx_ts_lab.explanation.coupling_graph import CouplingEdge, CouplingGraph

    graph = CouplingGraph(
        edges_by_target={
            "tgt": [CouplingEdge(source="src", target="tgt", weight=0.7, lag_steps=0)],
        },
        source="declared",
    )
    causes = walk_cascade(graph, "tgt", max_hops=2)
    assert len(causes) == 1
    assert causes[0].source_channel == "src"


def test_cascade_walker_multi_hop_decay() -> None:
    from dgx_ts_lab.explanation import walk_cascade
    from dgx_ts_lab.explanation.coupling_graph import CouplingEdge, CouplingGraph

    graph = CouplingGraph(
        edges_by_target={
            "B": [CouplingEdge(source="A", target="B", weight=0.9)],
            "C": [CouplingEdge(source="B", target="C", weight=0.8)],
        },
        source="declared",
    )
    causes = walk_cascade(graph, "C", max_hops=3, depth_decay=0.5)
    # Should find B (1-hop) and A (2-hop via B)
    names = [c.source_channel for c in causes]
    assert "B" in names
    assert "A" in names
    # B should rank above A (direct preferred)
    b_idx = names.index("B")
    a_idx = names.index("A")
    assert b_idx < a_idx


def test_cascade_walker_empty_graph_returns_nothing() -> None:
    from dgx_ts_lab.explanation import walk_cascade
    from dgx_ts_lab.explanation.coupling_graph import CouplingGraph

    causes = walk_cascade(CouplingGraph(), "anything", max_hops=2)
    assert causes == []


# ── Report writer ──────────────────────────────────────────────────────


def test_report_writer_emits_md_and_json(tmp_path: Path) -> None:
    from dgx_ts_lab.explanation import ExplanationReport, write_report
    from dgx_ts_lab.explanation.report_schema import Cause, ChannelAttribution

    report = ExplanationReport(
        detector_name="test_det",
        dataset_name="test_ds",
        window_idx=42,
        window_length=64,
        window_start_ms=12345000,
        anomaly_score=3.14,
        threshold=1.5,
        ranked_channels=[
            ChannelAttribution(channel_name="ch_a", score=1.0, rank=1, physics_covered=False),
            ChannelAttribution(channel_name="ch_b", score=0.5, rank=2, physics_covered=True),
        ],
        cascade=[
            Cause(source_channel="upstream_x", target_channel="ch_a", weight=0.7, lag_steps=2),
        ],
        coupling_source="declared",
        attribution_method="integrated_gradients",
    )
    md_path, json_path = write_report(report, tmp_path)
    assert md_path.exists()
    assert json_path.exists()
    # Markdown contains the channel + cascade tables
    md = md_path.read_text()
    assert "ch_a" in md
    assert "upstream_x" in md
    # JSON round-trips
    loaded = json.loads(json_path.read_text())
    assert loaded["window_idx"] == 42
    assert loaded["ranked_channels"][0]["channel_name"] == "ch_a"
    assert loaded["cascade"][0]["source_channel"] == "upstream_x"


# ── End-to-end: dataset → detector → attribute → cascade → report ──────


def test_end_to_end_explanation_pipeline(tmp_path: Path) -> None:
    from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset
    from dgx_ts_lab.explanation import (
        ExplanationReport,
        attribute_window,
        build_coupling_graph,
        walk_cascade,
        write_report,
    )
    from dgx_ts_lab.models.baseline import RollingMeanDetector

    ds = TrivialSyntheticDataset(n_samples=1000, n_channels=4, seed=0)
    det = RollingMeanDetector()
    det.fit(ds, FitMode.ZEROSHOT, {})

    win = next(ds.windows(length=64, stride=64))
    ranked = attribute_window(det, win, n_permutation_trials=2)
    graph = build_coupling_graph(ds, strategy="learned", correlation_threshold=0.0)
    cascade = walk_cascade(graph, ranked[0].channel_name, max_hops=2)

    report = ExplanationReport(
        detector_name=det.name,
        dataset_name=ds.name,
        window_idx=0,
        window_length=64,
        window_start_ms=int(win.timestamps[0]),
        anomaly_score=float(det.score(win).scores.max()),
        ranked_channels=ranked,
        cascade=cascade,
        coupling_source=graph.source,
        attribution_method="permutation",
    )
    md, js = write_report(report, tmp_path)
    assert md.exists() and js.exists()
    assert json.loads(js.read_text())["detector_name"] == "rolling_mean"
