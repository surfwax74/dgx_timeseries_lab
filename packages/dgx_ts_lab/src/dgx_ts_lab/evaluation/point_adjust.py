"""Point-adjusted F1 — the de facto standard metric for time-series AD with
collective anomalies.

Definition (Xu et al. 2018): if ANY point inside a true anomaly segment is
flagged, the WHOLE segment is treated as correctly detected.

This metric is generous compared to step-wise F1 but reflects the
operational reality: a single alarm somewhere inside an anomaly window is
usually sufficient for the on-call engineer to investigate.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _segments(labels: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return [(start, end)] (end exclusive) for contiguous True runs in labels."""
    if labels.size == 0:
        return []
    diff = np.diff(labels.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=False)]


def point_adjust(
    y_true: npt.NDArray[np.bool_],
    y_pred: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Apply point-adjust: for each true anomaly segment, if any pred is True
    inside the segment, set all preds inside the segment to True.

    Returns a new prediction array (does not modify input).
    """
    adjusted = y_pred.copy()
    for s, e in _segments(y_true):
        if y_pred[s:e].any():
            adjusted[s:e] = True
    return adjusted


def point_adjusted_metrics(
    y_true: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float32],
    threshold: float,
) -> dict[str, float]:
    """Precision / recall / F1 under point-adjustment at the given threshold."""
    from sklearn.metrics import precision_recall_fscore_support

    y_pred = scores > threshold
    y_pred_adj = point_adjust(y_true, y_pred)

    out = {"threshold_pa": float(threshold)}
    if y_true.sum() == 0:
        out.update(pa_precision=0.0, pa_recall=0.0, pa_f1=0.0)
        return out

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred_adj, average="binary", zero_division=0
    )
    out["pa_precision"] = float(p)
    out["pa_recall"] = float(r)
    out["pa_f1"] = float(f)
    return out
