"""Standard anomaly-detection metrics + threshold calibration.

Phase 0 uses a small subset (basic_metrics + percentile-based threshold).
Point-adjusted F1 and VUS arrive in Phase 2 alongside the bake-off CLI.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def calibrate_threshold(
    scores: npt.NDArray[np.float32],
    method: str = "percentile",
    percentile: float = 99.0,
) -> float:
    """Pick a binary-decision threshold from a score distribution.

    Currently supports unsupervised methods only — labeled F1-max calibration
    will be added when we have datasets where val labels are reliable.
    """
    if method == "percentile":
        return float(np.percentile(scores, percentile))
    if method == "mean_plus_3sigma":
        return float(scores.mean() + 3.0 * scores.std())
    raise ValueError(f"unknown calibration method: {method}")


def basic_metrics(
    y_true: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float32],
    threshold: float,
) -> dict[str, float]:
    """Precision, recall, F1, ROC-AUC, PR-AUC at the given threshold."""
    y_pred = scores > threshold
    out: dict[str, float] = {"threshold": float(threshold)}

    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        # Degenerate label set; AUC undefined. Return zeros / NaN.
        out.update(precision=0.0, recall=0.0, f1=0.0, roc_auc=float("nan"), pr_auc=float("nan"))
        return out

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out["precision"] = float(p)
    out["recall"] = float(r)
    out["f1"] = float(f)
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    except ValueError:
        out["pr_auc"] = float("nan")
    return out
