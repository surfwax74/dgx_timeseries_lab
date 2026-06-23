"""Visualization helpers for benchmark + bake-off results.

Consumes either:
    (a) raw (y_true, scores) arrays — single-run plots, or
    (b) a directory written by `dgx-ts benchmark` (benchmark_report.json
        + per-run *_scored.npz files) — comparison plots across detectors.

Every plot function:
    * accepts a matplotlib Axes (so notebooks can compose figures), or
    * returns a Figure when called standalone with `out_path=`.

All artifacts are presentation-ready: title, axis labels, legend,
gridlines, and a tight layout. PNG and SVG output supported (just change
the file extension).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe in CI / headless
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    _MATPLOTLIB_OK = True
except ImportError:                                  # pragma: no cover
    _MATPLOTLIB_OK = False
    Figure = Axes = object  # type: ignore[assignment,misc]


from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ── Loaders ──────────────────────────────────────────────────────────────


@dataclass
class ScoredRun:
    """One (detector, dataset, seed) result loaded for plotting."""

    detector: str
    dataset: str
    seed: int
    split: str                   # "val" | "test"
    scores: np.ndarray
    labels: np.ndarray
    val_roc_auc: float = float("nan")
    val_pr_auc: float = float("nan")
    val_f1: float = float("nan")
    test_roc_auc: float = float("nan")
    elapsed_s: float = 0.0

    @property
    def display_name(self) -> str:
        return f"{self.detector} (seed={self.seed})"


def load_benchmark_runs(
    benchmark_dir: str | Path,
    split: str = "val",
) -> list[ScoredRun]:
    """Load all (detector, dataset, seed) runs from a benchmark output dir.

    Reads `benchmark_report.json` for metrics + per-run npz files. Skips
    runs whose npz files are missing (graceful when benchmark.py was run
    against a detector that didn't expose val arrays).
    """
    bd = Path(benchmark_dir)
    rep_path = bd / "benchmark_report.json"
    if not rep_path.exists():
        raise FileNotFoundError(f"no benchmark_report.json under {bd}")
    rows = json.loads(rep_path.read_text())
    out: list[ScoredRun] = []
    for row in rows:
        if row.get("error"):
            continue
        det = row["detector"]
        ds = row["dataset"]
        seed = int(row["seed"])
        npz_path = bd / f"{det}__{ds}__s{seed}__{split}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        out.append(
            ScoredRun(
                detector=det,
                dataset=ds,
                seed=seed,
                split=split,
                scores=data["scores"].astype(np.float32),
                labels=data["labels"].astype(np.bool_),
                val_roc_auc=float(row.get("val_metrics", {}).get("roc_auc", float("nan"))),
                val_pr_auc=float(row.get("val_metrics", {}).get("pr_auc", float("nan"))),
                val_f1=float(row.get("val_metrics", {}).get("f1", float("nan"))),
                test_roc_auc=float(row.get("test_metrics", {}).get("roc_auc", float("nan"))),
                elapsed_s=float(row.get("elapsed_s", 0.0)),
            )
        )
    return out


# ── Single-run curves ────────────────────────────────────────────────────


def _ensure_mpl() -> None:
    if not _MATPLOTLIB_OK:
        raise RuntimeError("matplotlib is required for visualize.* functions")


def _safe_roc_curve(labels: np.ndarray, scores: np.ndarray):
    """ROC curve with NaN-safe fallback for degenerate label sets."""
    if labels.sum() == 0 or labels.sum() == labels.size:
        return (
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.0]),
        )
    return roc_curve(labels, scores)


def plot_roc_curves(
    runs: Sequence[ScoredRun],
    title: str = "ROC curves",
    out_path: str | Path | None = None,
    ax: Axes | None = None,
) -> Figure | None:
    """Overlay ROC curves for multiple runs. AUC printed in the legend."""
    _ensure_mpl()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=120)
    else:
        fig = ax.figure  # type: ignore[union-attr]
    for r in runs:
        fpr, tpr, _ = _safe_roc_curve(r.labels, r.scores)
        try:
            auc_val = roc_auc_score(r.labels, r.scores)
        except ValueError:
            auc_val = float("nan")
        ax.plot(fpr, tpr, lw=2, label=f"{r.detector} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    if standalone:
        fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig if standalone else None


def plot_pr_curves(
    runs: Sequence[ScoredRun],
    title: str = "Precision-Recall curves",
    out_path: str | Path | None = None,
    ax: Axes | None = None,
) -> Figure | None:
    """Overlay PR curves. Average-precision (PR-AUC) shown in legend."""
    _ensure_mpl()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=120)
    else:
        fig = ax.figure  # type: ignore[union-attr]
    for r in runs:
        if r.labels.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(r.labels, r.scores)
        try:
            ap = average_precision_score(r.labels, r.scores)
        except ValueError:
            ap = float("nan")
        ax.plot(recall, precision, lw=2, label=f"{r.detector} (AP={ap:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    if standalone:
        fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig if standalone else None


# ── Comparison / leaderboard plots ──────────────────────────────────────


def plot_auc_bar(
    runs: Sequence[ScoredRun],
    metric: str = "roc_auc",
    title: str | None = None,
    out_path: str | Path | None = None,
    ax: Axes | None = None,
) -> Figure | None:
    """Bar chart of AUC (or another scalar metric) per detector.

    Averages across seeds when there are multiple per detector and draws
    a black error bar at +/- 1 std.
    """
    _ensure_mpl()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    else:
        fig = ax.figure  # type: ignore[union-attr]

    # Group: (detector, dataset) → list of metric values across seeds
    groups: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        try:
            if metric == "roc_auc":
                v = roc_auc_score(r.labels, r.scores) if r.labels.sum() else float("nan")
            elif metric == "pr_auc":
                v = (
                    average_precision_score(r.labels, r.scores)
                    if r.labels.sum() else float("nan")
                )
            else:
                v = float("nan")
        except ValueError:
            v = float("nan")
        groups.setdefault((r.detector, r.dataset), []).append(v)

    # Sort: by mean metric descending
    items = sorted(
        groups.items(),
        key=lambda kv: -np.nanmean(kv[1]) if not all(np.isnan(kv[1])) else 1e9,
    )
    labels = [f"{det}\n{ds}" for (det, ds), _ in items]
    means = [float(np.nanmean(v)) for _, v in items]
    stds = [float(np.nanstd(v)) for _, v in items]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#4c72b0", alpha=0.85)
    for bar, m in zip(bars, means, strict=False):
        if not np.isnan(m):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{m:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_ylim(0, max(1.0, max(means) + 0.05) if means else 1.0)
    ax.set_title(title or f"{metric.replace('_', ' ').upper()} by detector × dataset")
    ax.grid(True, axis="y", alpha=0.3)
    if standalone:
        fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig if standalone else None


def plot_score_timeline(
    run: ScoredRun,
    threshold: float | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
    ax: Axes | None = None,
) -> Figure | None:
    """Per-step score trace, with anomaly windows shaded and threshold dashed.

    Best for the demo storyboard: 'detector fired here, label said yes here'.
    """
    _ensure_mpl()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10.0, 3.5), dpi=120)
    else:
        fig = ax.figure  # type: ignore[union-attr]

    t = np.arange(len(run.scores))
    ax.plot(t, run.scores, lw=1.0, color="#4c72b0", label="score")
    if threshold is not None:
        ax.axhline(threshold, color="#c44e52", lw=1, ls="--", label=f"threshold={threshold:.3g}")
    # Shade label-positive regions
    if run.labels.any():
        in_anom = False
        start = 0
        for i in range(len(run.labels)):
            if run.labels[i] and not in_anom:
                in_anom = True
                start = i
            elif not run.labels[i] and in_anom:
                ax.axvspan(start, i, color="#c44e52", alpha=0.15)
                in_anom = False
        if in_anom:
            ax.axvspan(start, len(run.labels), color="#c44e52", alpha=0.15)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Anomaly score")
    ax.set_title(title or f"Score trace — {run.display_name} on {run.dataset}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    if standalone:
        fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig if standalone else None


def plot_parameter_sweep(
    param_name: str,
    param_values: Sequence[float | int],
    runs_by_value: dict,
    metric: str = "roc_auc",
    title: str | None = None,
    out_path: str | Path | None = None,
    ax: Axes | None = None,
) -> Figure | None:
    """Plot metric vs. a swept parameter for one or more detectors.

    `runs_by_value`: dict[detector_name, list[ScoredRun]] where index i
    corresponds to param_values[i]. Caller is responsible for ordering.
    """
    _ensure_mpl()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=120)
    else:
        fig = ax.figure  # type: ignore[union-attr]
    for detector, runs in runs_by_value.items():
        ys: list[float] = []
        for r in runs:
            try:
                if metric == "roc_auc":
                    v = roc_auc_score(r.labels, r.scores) if r.labels.sum() else float("nan")
                elif metric == "pr_auc":
                    v = average_precision_score(r.labels, r.scores) if r.labels.sum() else float("nan")
                else:
                    v = float("nan")
            except ValueError:
                v = float("nan")
            ys.append(v)
        ax.plot(param_values, ys, "o-", lw=2, markersize=6, label=detector)
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(title or f"{metric.upper()} vs. {param_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    if standalone:
        fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig if standalone else None


# ── One-call report bundle ──────────────────────────────────────────────


def render_benchmark_report_figures(
    benchmark_dir: str | Path,
    output_dir: str | Path,
    splits: Iterable[str] = ("val", "test"),
    formats: Iterable[str] = ("png",),
) -> list[Path]:
    """Generate the full set of presentation-grade figures from a benchmark dir.

    Writes one ROC overlay + one PR overlay per (dataset, split), and a
    single AUC bar comparing all detectors on val.

    Returns the list of written file paths.
    """
    _ensure_mpl()
    bd = Path(benchmark_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    auc_runs: list[ScoredRun] = []
    for split in splits:
        runs = load_benchmark_runs(bd, split=split)
        if not runs:
            continue
        if split == "val":
            auc_runs = runs
        # Group by dataset
        by_ds: dict[str, list[ScoredRun]] = {}
        for r in runs:
            by_ds.setdefault(r.dataset, []).append(r)
        for ds, ds_runs in by_ds.items():
            for ext in formats:
                roc_path = out / f"roc__{ds}__{split}.{ext}"
                pr_path = out / f"pr__{ds}__{split}.{ext}"
                plot_roc_curves(
                    ds_runs,
                    title=f"ROC — {ds} ({split})",
                    out_path=roc_path,
                )
                plot_pr_curves(
                    ds_runs,
                    title=f"PR — {ds} ({split})",
                    out_path=pr_path,
                )
                written.extend([roc_path, pr_path])

    if auc_runs:
        for ext in formats:
            bar_path = out / f"auc_bar_val.{ext}"
            plot_auc_bar(
                auc_runs,
                metric="roc_auc",
                title="Validation ROC-AUC by detector × dataset",
                out_path=bar_path,
            )
            written.append(bar_path)

    if _MATPLOTLIB_OK:
        plt.close("all")
    return written
