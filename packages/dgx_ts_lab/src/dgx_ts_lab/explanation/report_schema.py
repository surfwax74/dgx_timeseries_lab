"""Typed schema for the explanation report.

Single source of truth for the JSON sibling. Phase 11's RAG retrieval
parses this schema directly; the Markdown report is a human rendering of
the same content.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ChannelAttribution:
    """Per-channel contribution to the anomaly score for one window."""

    channel_name: str
    score: float                # raw attribution magnitude
    rank: int                   # 1 = most contributing
    physics_covered: bool = False    # if PINN-wrapped: was this explained by physics?


@dataclass
class Cause:
    """An upstream channel suspected of driving an anomaly in a target channel."""

    source_channel: str
    target_channel: str
    weight: float              # graph edge weight (correlation or gain)
    lag_steps: int = 0
    via: str = ""              # path description if multi-hop, else empty


@dataclass
class ExplanationReport:
    """One window's full explanation."""

    detector_name: str
    dataset_name: str
    window_idx: int
    window_length: int
    window_start_ms: int

    anomaly_score: float
    threshold: float | None = None

    # Top-K per-channel attributions, rank 1 first
    ranked_channels: list[ChannelAttribution] = field(default_factory=list)
    attribution_method: str = "unknown"

    # Upstream root-cause chain (deepest hop first)
    cascade: list[Cause] = field(default_factory=list)
    coupling_source: str = "unknown"     # "declared" | "learned" | "none"

    # Optional fields populated when extra info is available
    fault_type_predicted: str | None = None       # from Phase 6 multi-task head
    physics_model_name: str | None = None         # for PINN-wrapped detectors
    plot_paths: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    # Provenance
    run_id: str | None = None
    detector_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
