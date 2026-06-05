"""Phase 7 — detector-agnostic explanation layer.

When AD fires, this package turns the raw anomaly score into a structured
explanation: which channels drove the score, which UPSTREAM channels are
the likely root cause (via the layered-synth coupling graph or a learned
correlation graph), and a polished Markdown + JSON report ready for
operators (and for Phase 11's LLM ops co-pilot to consume).

Public API:

    attribute_window(detector, window) -> ChannelAttribution
    walk_cascade(graph, anomalous_channel) -> list[Cause]
    write_report(report, out_dir) -> tuple[Path, Path]    # (.md, .json)
    build_coupling_graph(dataset, strategy="auto") -> CouplingGraph

CLI:

    dgx-ts explain dataset=... model=... +checkpoint=... +window_idx=42
"""

from .attribution import ChannelAttribution, attribute_window
from .cascade_walker import Cause, walk_cascade
from .coupling_graph import CouplingGraph, build_coupling_graph
from .report_schema import ExplanationReport
from .report_writer import write_report

__all__ = [
    "Cause",
    "ChannelAttribution",
    "CouplingGraph",
    "ExplanationReport",
    "attribute_window",
    "build_coupling_graph",
    "walk_cascade",
    "write_report",
]
