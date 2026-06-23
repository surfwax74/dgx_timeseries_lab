"""Dual-source channel coupling graph for the cascade walker.

Two construction paths, selected automatically by ``build_coupling_graph``:

    1. **Declared** (preferred): if the dataset is a LayeredSyntheticDataset
       with introspectable L3 coupling components, extract the ground-truth
       directed graph (source channel → target channel with gain + lag).
    2. **Learned**: walk training windows, compute per-channel pair Pearson
       correlations (also at small lags). Edges retained where
       |correlation| > threshold.

Always falls back gracefully — a missing or empty graph just means the
cascade walker returns no upstream causes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from dgx_ts_core.data import TelemetryDataset


@dataclass
class CouplingEdge:
    source: str
    target: str
    weight: float         # gain (declared) or |correlation| (learned)
    lag_steps: int = 0


@dataclass
class CouplingGraph:
    """Directed graph: target_channel → list of (source, weight, lag)."""

    edges_by_target: dict[str, list[CouplingEdge]] = field(default_factory=dict)
    source: str = "none"   # "declared" | "learned" | "none"

    def upstream(self, channel: str, top_k: int | None = None) -> list[CouplingEdge]:
        edges = self.edges_by_target.get(channel, [])
        edges = sorted(edges, key=lambda e: -abs(e.weight))
        if top_k is not None:
            edges = edges[:top_k]
        return edges

    def n_edges(self) -> int:
        return sum(len(v) for v in self.edges_by_target.values())


# ── Construction strategies ────────────────────────────────────────────


def _try_declared(dataset: TelemetryDataset) -> CouplingGraph | None:
    """Extract a declared graph from a LayeredSyntheticDataset's components."""
    components = getattr(dataset, "_components", None)
    if not components:
        return None
    edges: dict[str, list[CouplingEdge]] = {}

    for comp in components:
        kind = getattr(comp, "kind", "")
        if kind == "linear_coupling":
            tgt = comp.target
            edges.setdefault(tgt, []).append(
                CouplingEdge(
                    source=comp.source, target=tgt,
                    weight=float(comp.gain), lag_steps=int(comp.lag_steps),
                )
            )
        elif kind == "inverse_coupling":
            tgt = comp.target
            edges.setdefault(tgt, []).append(
                CouplingEdge(
                    source=comp.source, target=tgt,
                    weight=-float(comp.gain), lag_steps=0,
                )
            )
        elif kind == "sum_coupling":
            tgt = comp.target
            for src, gain in zip(comp.sources, comp.gains, strict=False):
                edges.setdefault(tgt, []).append(
                    CouplingEdge(source=src, target=tgt, weight=float(gain), lag_steps=0)
                )
    if not edges:
        return None
    return CouplingGraph(edges_by_target=edges, source="declared")


def _learn_from_correlations(
    dataset: TelemetryDataset,
    n_samples: int = 4096,
    correlation_threshold: float = 0.3,
    max_lag: int = 0,
) -> CouplingGraph:
    """Build a graph from cross-channel Pearson correlations on training data.

    For ``max_lag > 0``, also computes lagged correlations target[t] vs
    source[t-lag] for lag ∈ [1, max_lag] and keeps the best-magnitude lag
    per (source, target) pair.
    """
    if hasattr(dataset, "_data"):
        data = dataset._data
    else:
        chunks = []
        for w in dataset.windows(length=1024, stride=1024):
            chunks.append(w.tensor)
        if not chunks:
            return CouplingGraph(source="learned")
        data = np.concatenate(chunks, axis=0)

    if data.shape[0] > n_samples:
        # Sample evenly
        idx = np.linspace(0, data.shape[0] - 1, n_samples).astype(np.int64)
        data = data[idx]

    channel_names = [ch.name for ch in dataset.channels]
    C = data.shape[1]
    edges: dict[str, list[CouplingEdge]] = {}

    # Standardize once for fast correlation
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    z = (data - mean) / std

    for target_i in range(C):
        target_name = channel_names[target_i]
        for source_i in range(C):
            if source_i == target_i:
                continue
            source_name = channel_names[source_i]
            best_corr = 0.0
            best_lag = 0
            # Compute correlation at lag 0 + small lags
            for lag in range(0, max_lag + 1):
                if lag == 0:
                    corr = float(np.mean(z[:, target_i] * z[:, source_i]))
                else:
                    if lag >= len(z):
                        continue
                    corr = float(np.mean(z[lag:, target_i] * z[:-lag, source_i]))
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            if abs(best_corr) >= correlation_threshold:
                edges.setdefault(target_name, []).append(
                    CouplingEdge(
                        source=source_name,
                        target=target_name,
                        weight=best_corr,
                        lag_steps=best_lag,
                    )
                )

    return CouplingGraph(edges_by_target=edges, source="learned")


def build_coupling_graph(
    dataset: TelemetryDataset,
    strategy: str = "auto",
    correlation_threshold: float = 0.3,
    max_lag: int = 0,
    n_samples: int = 4096,
) -> CouplingGraph:
    """Build a coupling graph using the requested strategy.

    Strategies:
        "auto"      — declared if available, else learned
        "declared"  — only declared (returns empty graph if not available)
        "learned"   — only learned from correlations
        "none"      — empty graph
    """
    if strategy == "none":
        return CouplingGraph(source="none")
    if strategy in ("auto", "declared"):
        declared = _try_declared(dataset)
        if declared is not None:
            return declared
        if strategy == "declared":
            return CouplingGraph(source="declared")
    return _learn_from_correlations(
        dataset,
        n_samples=n_samples,
        correlation_threshold=correlation_threshold,
        max_lag=max_lag,
    )
