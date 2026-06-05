"""Per-channel attribution for detector outputs.

Captum-backed for differentiable detectors (those with ``.module``), with
permutation-importance fallback for classical detectors (rolling_mean,
operator_fingerprint without gradient path, etc.).

The returned ``ChannelAttribution`` is normalized so the top-1 channel has
score 1.0; subsequent channels are relative magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dgx_ts_core.data import TelemetryWindow

from .report_schema import ChannelAttribution


@dataclass
class AttributionResult:
    """Raw attribution output before ranking + normalization."""

    per_channel_scores: np.ndarray   # (C,) magnitude per channel
    method: str                       # e.g. "integrated_gradients", "permutation"


# ── Public API ─────────────────────────────────────────────────────────


def attribute_window(
    detector,
    window: TelemetryWindow,
    top_k: int | None = None,
    n_steps_ig: int = 32,
    n_permutation_trials: int = 5,
) -> list[ChannelAttribution]:
    """Return ranked per-channel attribution for a single window.

    Dispatches by detector capability:
        - if detector exposes ``module`` + ``compute_score_batch`` (differentiable
          neural detector), try Captum Integrated Gradients
        - else fall back to permutation importance using ``score()``
    """
    raw = _try_integrated_gradients(detector, window, n_steps=n_steps_ig)
    if raw is None:
        raw = _permutation_importance(
            detector, window, n_trials=n_permutation_trials
        )
    return _rank_and_normalize(raw, window, top_k=top_k)


# ── Captum IG path ─────────────────────────────────────────────────────


def _try_integrated_gradients(
    detector, window: TelemetryWindow, n_steps: int
) -> AttributionResult | None:
    if not hasattr(detector, "module") or detector.module is None:
        return None
    if not hasattr(detector, "compute_score_batch"):
        return None
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        return None

    device = next(detector.module.parameters()).device
    x = torch.from_numpy(window.tensor).float().unsqueeze(0).to(device)

    def _score_scalar(x_in: torch.Tensor) -> torch.Tensor:
        # Captum needs a scalar (or per-sample scalar) output. We use the
        # max per-step score per sample as the target — proxies "how
        # anomalous is the most anomalous timestep in this window".
        scores = detector.compute_score_batch({"x": x_in})   # (B, T)
        return scores.max(dim=-1).values                     # (B,)

    try:
        ig = IntegratedGradients(_score_scalar)
        # Baseline = zeros (typical for IG); the integration path is linear from baseline to x
        baseline = torch.zeros_like(x)
        attributions = ig.attribute(x, baselines=baseline, n_steps=n_steps)
        # Sum |attribution| over time to get per-channel magnitude
        per_channel = attributions.abs().sum(dim=(0, 1)).detach().cpu().numpy().astype(np.float32)
        return AttributionResult(per_channel_scores=per_channel, method="integrated_gradients")
    except Exception:
        # Captum can fail on exotic models (multi-output, dynamic shapes); fall through.
        return None


# ── Permutation importance fallback ────────────────────────────────────


def _permutation_importance(
    detector, window: TelemetryWindow, n_trials: int
) -> AttributionResult:
    """For each channel, replace with shuffled values and measure score drop.

    The MAGNITUDE of the score change is the per-channel importance — high
    means perturbing this channel meaningfully changes the anomaly score.
    """
    rng = np.random.default_rng(0)
    T, C = window.tensor.shape
    base_score = float(detector.score(window).scores.max())
    importance = np.zeros(C, dtype=np.float32)
    for c in range(C):
        deltas: list[float] = []
        for _ in range(n_trials):
            perturbed = window.tensor.copy()
            # Shuffle this channel across time
            perm = rng.permutation(T)
            perturbed[:, c] = perturbed[perm, c]
            perturbed_window = TelemetryWindow(
                tensor=perturbed,
                timestamps=window.timestamps,
                channels=window.channels,
                labels=window.labels,
                mask=window.mask,
                provenance=window.provenance,
            )
            new_score = float(detector.score(perturbed_window).scores.max())
            deltas.append(abs(new_score - base_score))
        importance[c] = float(np.mean(deltas))
    return AttributionResult(per_channel_scores=importance, method="permutation")


# ── Ranking + normalization ────────────────────────────────────────────


def _rank_and_normalize(
    raw: AttributionResult,
    window: TelemetryWindow,
    top_k: int | None,
) -> list[ChannelAttribution]:
    scores = raw.per_channel_scores
    if scores.size == 0 or scores.max() <= 0:
        # No signal — return uniformly-ranked channels with score 0
        ranked = list(range(len(window.channels)))
        return [
            ChannelAttribution(
                channel_name=window.channels[i].name,
                score=0.0,
                rank=i + 1,
            )
            for i in ranked
        ]
    # Normalize to [0, 1] with top-1 = 1.0
    norm = scores / max(float(scores.max()), 1e-12)
    order = np.argsort(-norm).tolist()
    out: list[ChannelAttribution] = []
    for rank, i in enumerate(order, start=1):
        out.append(
            ChannelAttribution(
                channel_name=window.channels[i].name,
                score=float(norm[i]),
                rank=rank,
            )
        )
    if top_k is not None:
        out = out[:top_k]
    return out
