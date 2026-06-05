"""Matplotlib visualizations for explanation reports.

Renders two artifacts per report:

    score_timeline.png      per-step anomaly score with the window highlighted
    channel_attribution.png horizontal bar chart of top-K channel attributions

These land in the MLflow run's artifacts dir alongside the .md + .json.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Force a non-interactive backend so this works in headless / CI / DGX runs.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402  must come after backend set

from .report_schema import ChannelAttribution


def render_score_timeline(
    scores: np.ndarray,
    window_idx_range: tuple[int, int],
    threshold: float | None,
    out_path: Path,
) -> Path:
    """Plot the full score trace with the explained window highlighted."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(scores, color="steelblue", linewidth=0.7, label="anomaly score")
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=0.5, label=f"threshold={threshold:.3f}")
    start, end = window_idx_range
    ax.axvspan(start, end, color="orange", alpha=0.3, label="explained window")
    ax.set_xlabel("step")
    ax.set_ylabel("anomaly score")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Anomaly score timeline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_channel_attribution(
    ranked_channels: list[ChannelAttribution],
    out_path: Path,
    top_k: int = 15,
) -> Path:
    """Horizontal bar chart of top-K channels by attribution score."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top = ranked_channels[:top_k]
    names = [c.channel_name for c in top][::-1]
    vals = [c.score for c in top][::-1]
    colors = ["lightcoral" if c.physics_covered else "steelblue" for c in top][::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(top))))
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("normalized attribution")
    ax.set_title(f"Top-{len(top)} channel attributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
