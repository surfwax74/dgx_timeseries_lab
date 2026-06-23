"""Tool implementations the co-pilot exposes to the LLM.

Each tool has two parts:
    (1) a `ToolDef` describing its JSONSchema, passed to the LLM
    (2) a Python callable `(args_dict) -> str` that executes it

A `ToolRegistry` bundles both — the orchestrator gives the registry
a `ToolCall`, gets back a string result it then echoes back to the LLM.

Built-in tools:
    query_telemetry         summary stats over a channel + time window
    query_anomaly_history   recent anomalies above a threshold
    lookup_procedure        RAG over the mission-procedure corpus
    read_model_card         dump the deployed detector's Phase 5 model card

All tools accept a `context` injected by the registry — typically the
dataset, recent scores, RAG index, and detector handle.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .backend import ToolCall, ToolDef
from .rag import CosineRAGIndex

# ── ToolRegistry ─────────────────────────────────────────────────────────


@dataclass
class CopilotContext:
    """Bundle of artifacts the tools may read. Optional fields — tools
    that need a missing piece return an error string the LLM can read."""

    telemetry: np.ndarray | None = None              # (T, C)
    channel_names: list[str] = field(default_factory=list)
    sample_rate_hz: float = 1.0
    anomaly_scores: np.ndarray | None = None         # (T,)
    anomaly_threshold: float | None = None
    rag_index: CosineRAGIndex | None = None
    model_card_path: str | Path | None = None


ToolFn = Callable[[dict[str, Any], CopilotContext], str]


@dataclass
class ToolRegistry:
    """Maps tool name → (ToolDef, ToolFn)."""

    _defs: dict[str, ToolDef] = field(default_factory=dict)
    _fns: dict[str, ToolFn] = field(default_factory=dict)
    context: CopilotContext = field(default_factory=CopilotContext)

    def register(self, tool_def: ToolDef, fn: ToolFn) -> None:
        self._defs[tool_def.name] = tool_def
        self._fns[tool_def.name] = fn

    def list_defs(self) -> list[ToolDef]:
        return list(self._defs.values())

    def invoke(self, call: ToolCall) -> str:
        fn = self._fns.get(call.name)
        if fn is None:
            return f"ERROR: unknown tool {call.name!r}"
        try:
            return fn(call.arguments or {}, self.context)
        except Exception as e:                                  # noqa: BLE001
            return f"ERROR: tool {call.name!r} raised {type(e).__name__}: {e}"


# ── Built-in tools ───────────────────────────────────────────────────────


_QUERY_TELEMETRY_DEF = ToolDef(
    name="query_telemetry",
    description=(
        "Summary statistics for a telemetry channel over a time window. "
        "Returns mean, std, min, max, and last value as JSON. Use this "
        "when the operator asks about a specific channel's behavior."
    ),
    parameters={
        "type": "object",
        "properties": {
            "channel": {"type": "string", "description": "Channel name"},
            "start_step": {
                "type": "integer",
                "description": "Start time-step index (inclusive). Negative counts back from end.",
            },
            "end_step": {
                "type": "integer",
                "description": "End time-step index (exclusive). Use 0 for 'until end'.",
            },
        },
        "required": ["channel"],
    },
)


def _query_telemetry(args: dict[str, Any], ctx: CopilotContext) -> str:
    if ctx.telemetry is None or not ctx.channel_names:
        return "ERROR: no telemetry available in context"
    ch = args.get("channel")
    if ch not in ctx.channel_names:
        return f"ERROR: unknown channel {ch!r}. Available: {ctx.channel_names[:20]}"
    col = ctx.channel_names.index(ch)
    T = ctx.telemetry.shape[0]
    start = int(args.get("start_step", 0))
    end = int(args.get("end_step", 0))
    if start < 0:
        start = max(0, T + start)
    if end <= 0:
        end = T
    end = min(end, T)
    if start >= end:
        return f"ERROR: empty window start={start} end={end}"
    seg = ctx.telemetry[start:end, col]
    return json.dumps(
        {
            "channel": ch,
            "n_samples": int(seg.size),
            "mean": float(seg.mean()),
            "std": float(seg.std()),
            "min": float(seg.min()),
            "max": float(seg.max()),
            "last": float(seg[-1]),
        }
    )


_QUERY_ANOMALY_HISTORY_DEF = ToolDef(
    name="query_anomaly_history",
    description=(
        "Returns the top-K time-steps where the anomaly score exceeded the "
        "configured threshold. Use this to ask 'when did anomalies fire?'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "top_k": {"type": "integer", "default": 10},
            "min_score": {"type": "number", "description": "Override threshold"},
        },
        "required": [],
    },
)


def _query_anomaly_history(args: dict[str, Any], ctx: CopilotContext) -> str:
    if ctx.anomaly_scores is None:
        return "ERROR: no anomaly scores in context"
    threshold = float(args.get("min_score") or ctx.anomaly_threshold or 0.0)
    top_k = int(args.get("top_k", 10))
    scores = ctx.anomaly_scores
    above = np.argwhere(scores >= threshold).reshape(-1)
    if above.size == 0:
        return json.dumps(
            {"threshold": threshold, "n_above_threshold": 0, "events": []}
        )
    ordered = above[np.argsort(-scores[above])[:top_k]]
    events = [
        {"step": int(i), "score": float(scores[i])}
        for i in ordered
    ]
    return json.dumps(
        {
            "threshold": threshold,
            "n_above_threshold": int(above.size),
            "events": events,
        }
    )


_LOOKUP_PROCEDURE_DEF = ToolDef(
    name="lookup_procedure",
    description=(
        "Search the mission-procedure RAG corpus for the most relevant "
        "documents to a natural-language query. Returns top-K chunks."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 3},
        },
        "required": ["query"],
    },
)


def _lookup_procedure(args: dict[str, Any], ctx: CopilotContext) -> str:
    if ctx.rag_index is None or ctx.rag_index.n_docs == 0:
        return "ERROR: no procedure index loaded in context"
    hits = ctx.rag_index.query(args.get("query", ""), top_k=int(args.get("top_k", 3)))
    return json.dumps(
        {
            "hits": [
                {
                    "doc_id": h.document.doc_id,
                    "title": h.document.title,
                    "score": h.score,
                    "text": h.document.text[:1500],
                }
                for h in hits
            ]
        }
    )


_READ_MODEL_CARD_DEF = ToolDef(
    name="read_model_card",
    description=(
        "Read the deployed detector's exported model card "
        "(Phase 5 artifact). Returns the YAML contents as a string."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
)


def _read_model_card(args: dict[str, Any], ctx: CopilotContext) -> str:
    if ctx.model_card_path is None:
        return "ERROR: no model_card_path in context"
    p = Path(ctx.model_card_path)
    if not p.exists():
        return f"ERROR: model card not found at {p}"
    return p.read_text(encoding="utf-8")


# ── Helpers ──────────────────────────────────────────────────────────────


def default_tool_registry(context: CopilotContext | None = None) -> ToolRegistry:
    """A pre-populated registry with all four standard tools."""
    reg = ToolRegistry(context=context or CopilotContext())
    reg.register(_QUERY_TELEMETRY_DEF, _query_telemetry)
    reg.register(_QUERY_ANOMALY_HISTORY_DEF, _query_anomaly_history)
    reg.register(_LOOKUP_PROCEDURE_DEF, _lookup_procedure)
    reg.register(_READ_MODEL_CARD_DEF, _read_model_card)
    return reg
