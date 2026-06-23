"""Tests for evaluation/visualize.py and the dgx-ts viz CLI pipeline.

These tests build synthetic (scores, labels) arrays + a fake benchmark
output dir, then assert that:
    * each plot function returns a Figure (when standalone) and writes
      a non-empty PNG to disk
    * load_benchmark_runs reconstructs ScoredRun objects from .npz files
    * render_benchmark_report_figures emits the expected file layout
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from dgx_ts_lab.evaluation.visualize import (
    ScoredRun,
    load_benchmark_runs,
    plot_auc_bar,
    plot_parameter_sweep,
    plot_pr_curves,
    plot_roc_curves,
    plot_score_timeline,
    render_benchmark_report_figures,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_scored_run(detector: str, n: int = 500, anomaly_frac: float = 0.05,
                     signal_strength: float = 2.0, seed: int = 0) -> ScoredRun:
    """Synthetic scores + labels — gaussian noise with elevated peaks at anomalies."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=np.bool_)
    n_anom = max(1, int(n * anomaly_frac))
    anom_idx = rng.choice(n, size=n_anom, replace=False)
    labels[anom_idx] = True
    scores = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    scores[anom_idx] += signal_strength
    return ScoredRun(
        detector=detector,
        dataset="synth",
        seed=seed,
        split="val",
        scores=scores,
        labels=labels,
    )


@pytest.fixture
def two_runs() -> list[ScoredRun]:
    return [
        _make_scored_run("detector_a", signal_strength=3.0, seed=0),
        _make_scored_run("detector_b", signal_strength=1.0, seed=0),
    ]


@pytest.fixture
def fake_benchmark_dir(tmp_path: Path) -> Path:
    """Fabricate a benchmark output directory matching `dgx-ts benchmark`'s shape."""
    bd = tmp_path / "fake_bench"
    bd.mkdir()
    # Two detectors × one dataset × two seeds
    rows = []
    for det, strength in [("det_strong", 3.0), ("det_weak", 1.0)]:
        for seed in (0, 1):
            run = _make_scored_run(det, signal_strength=strength, seed=seed)
            np.savez(
                bd / f"{det}__synth__s{seed}__val.npz",
                scores=run.scores,
                labels=run.labels,
            )
            np.savez(
                bd / f"{det}__synth__s{seed}__test.npz",
                scores=run.scores,
                labels=run.labels,
            )
            rows.append(
                {
                    "detector": det,
                    "dataset": "synth",
                    "seed": seed,
                    "val_metrics": {
                        "roc_auc": 0.95 if det == "det_strong" else 0.65,
                        "pr_auc": 0.85 if det == "det_strong" else 0.4,
                        "f1": 0.8 if det == "det_strong" else 0.3,
                    },
                    "test_metrics": {"roc_auc": 0.93 if det == "det_strong" else 0.6},
                    "elapsed_s": 1.5,
                    "error": None,
                }
            )
    (bd / "benchmark_report.json").write_text(json.dumps(rows, indent=2))
    return bd


# ── Single plot helpers ─────────────────────────────────────────────────


def test_plot_roc_curves_writes_png(tmp_path: Path, two_runs: list[ScoredRun]) -> None:
    out = tmp_path / "roc.png"
    fig = plot_roc_curves(two_runs, out_path=out)
    assert fig is not None
    assert out.exists() and out.stat().st_size > 1000  # non-empty PNG


def test_plot_pr_curves_writes_png(tmp_path: Path, two_runs: list[ScoredRun]) -> None:
    out = tmp_path / "pr.png"
    plot_pr_curves(two_runs, out_path=out)
    assert out.exists() and out.stat().st_size > 1000


def test_plot_auc_bar_with_multiple_seeds(tmp_path: Path) -> None:
    runs = [
        _make_scored_run("a", signal_strength=2.5, seed=s) for s in range(3)
    ] + [
        _make_scored_run("b", signal_strength=0.5, seed=s) for s in range(3)
    ]
    out = tmp_path / "auc.png"
    plot_auc_bar(runs, out_path=out)
    assert out.exists() and out.stat().st_size > 1000


def test_plot_score_timeline_with_threshold(tmp_path: Path) -> None:
    run = _make_scored_run("a", signal_strength=3.0, seed=0)
    out = tmp_path / "trace.png"
    plot_score_timeline(run, threshold=1.5, out_path=out)
    assert out.exists() and out.stat().st_size > 1000


def test_plot_parameter_sweep(tmp_path: Path) -> None:
    out = tmp_path / "sweep.png"
    values = [1.0, 2.0, 3.0]
    runs_by = {
        "det_a": [
            _make_scored_run("det_a", signal_strength=v, seed=0) for v in values
        ],
    }
    plot_parameter_sweep(
        param_name="signal_strength",
        param_values=values,
        runs_by_value=runs_by,
        out_path=out,
    )
    assert out.exists() and out.stat().st_size > 1000


# ── Loader + report bundle ──────────────────────────────────────────────


def test_load_benchmark_runs_reads_npz(fake_benchmark_dir: Path) -> None:
    runs = load_benchmark_runs(fake_benchmark_dir, split="val")
    assert len(runs) == 4   # 2 detectors × 2 seeds
    dets = sorted({r.detector for r in runs})
    assert dets == ["det_strong", "det_weak"]
    # ROC-AUC from the metrics block should be present
    assert runs[0].val_roc_auc in (0.95, 0.65)


def test_load_benchmark_runs_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_benchmark_runs(tmp_path / "nope", split="val")


def test_load_benchmark_runs_skips_errored_rows(tmp_path: Path) -> None:
    bd = tmp_path / "errored"
    bd.mkdir()
    (bd / "benchmark_report.json").write_text(
        json.dumps([{"detector": "x", "dataset": "y", "seed": 0, "error": "boom"}])
    )
    assert load_benchmark_runs(bd, split="val") == []


def test_render_benchmark_report_figures_emits_full_set(
    tmp_path: Path,
    fake_benchmark_dir: Path,
) -> None:
    out = tmp_path / "figures"
    written = render_benchmark_report_figures(
        benchmark_dir=fake_benchmark_dir,
        output_dir=out,
        splits=("val", "test"),
        formats=("png",),
    )
    assert len(written) >= 5
    names = {p.name for p in written}
    # ROC + PR for each split, plus auc bar
    assert "roc__synth__val.png" in names
    assert "pr__synth__val.png" in names
    assert "roc__synth__test.png" in names
    assert "pr__synth__test.png" in names
    assert "auc_bar_val.png" in names
    for p in written:
        assert p.exists() and p.stat().st_size > 1000


def test_render_benchmark_report_figures_svg_format(
    tmp_path: Path,
    fake_benchmark_dir: Path,
) -> None:
    out = tmp_path / "figures"
    written = render_benchmark_report_figures(
        benchmark_dir=fake_benchmark_dir,
        output_dir=out,
        splits=("val",),
        formats=("svg",),
    )
    svgs = [p for p in written if p.suffix == ".svg"]
    assert len(svgs) >= 3
    for p in svgs:
        text = p.read_text()
        assert text.startswith("<?xml") or text.startswith("<svg")


# ── Edge cases ──────────────────────────────────────────────────────────


def test_plot_roc_handles_degenerate_labels(tmp_path: Path) -> None:
    """All-zero labels: function should still draw a chance line and exit OK."""
    run = ScoredRun(
        detector="all_zero",
        dataset="synth",
        seed=0,
        split="val",
        scores=np.random.rand(100).astype(np.float32),
        labels=np.zeros(100, dtype=np.bool_),
    )
    out = tmp_path / "roc_degenerate.png"
    plot_roc_curves([run], out_path=out)
    assert out.exists() and out.stat().st_size > 500


def test_plot_pr_skips_runs_with_no_positives(tmp_path: Path) -> None:
    run = ScoredRun(
        detector="all_zero",
        dataset="synth",
        seed=0,
        split="val",
        scores=np.random.rand(100).astype(np.float32),
        labels=np.zeros(100, dtype=np.bool_),
    )
    out = tmp_path / "pr_degenerate.png"
    plot_pr_curves([run], out_path=out)
    assert out.exists()   # produced an (empty) plot
