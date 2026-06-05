"""Cascade walker — traces upstream causes for an anomalous channel.

Takes a CouplingGraph + a target channel + (optionally) a list of anomalous
channels with their attribution scores, and produces a ranked list of
``Cause`` entries explaining likely upstream root causes.

Implementation: BFS up to ``max_hops`` levels deep. Each hop's edges
contribute to a "causal score" = edge_weight * (1 / depth_penalty). Returns
sorted by absolute causal score descending.
"""

from __future__ import annotations

from .coupling_graph import CouplingGraph
from .report_schema import Cause


def walk_cascade(
    graph: CouplingGraph,
    target_channel: str,
    max_hops: int = 2,
    top_k: int | None = 5,
    depth_decay: float = 0.5,
) -> list[Cause]:
    """Return ranked upstream causes for the target channel.

    Multi-hop: a 2-hop cause's weight is multiplied by ``depth_decay``
    (default 0.5) so direct upstream is preferred over indirect.
    """
    causes: list[Cause] = []
    visited: set[str] = {target_channel}
    # BFS frontier: list of (current_channel, accumulated_weight, depth, path_str)
    frontier: list[tuple[str, float, int, str]] = [(target_channel, 1.0, 0, target_channel)]

    while frontier:
        next_frontier: list[tuple[str, float, int, str]] = []
        for current, acc_w, depth, path in frontier:
            if depth >= max_hops:
                continue
            for edge in graph.upstream(current):
                if edge.source in visited:
                    continue
                effective_w = acc_w * edge.weight * (depth_decay ** depth)
                via = path if depth > 0 else ""
                causes.append(
                    Cause(
                        source_channel=edge.source,
                        target_channel=target_channel,
                        weight=float(effective_w),
                        lag_steps=int(edge.lag_steps),
                        via=via,
                    )
                )
                visited.add(edge.source)
                next_frontier.append(
                    (edge.source, effective_w, depth + 1, f"{edge.source} -> {path}")
                )
        frontier = next_frontier

    causes.sort(key=lambda c: -abs(c.weight))
    if top_k is not None:
        causes = causes[:top_k]
    return causes
