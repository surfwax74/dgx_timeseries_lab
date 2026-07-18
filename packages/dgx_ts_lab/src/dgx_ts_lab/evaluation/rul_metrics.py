"""RUL (Remaining Useful Life) metrics.

These metrics assume **synthetic run-to-failure trajectories** with known
EOL — see docs/forecasting_rul_bakeoff.md for why we don't apply RUL to
real satellite telemetry (no run-to-failure labels in NASA Telemanom or
OPS-SAT). If C-MAPSS-style data is later ingested, these metrics apply
unchanged.

Shape convention:
    y_true_rul:  (N,)  actual remaining life at each test point, in the
                       time unit the scenario uses (typically days).
    y_pred_rul:  (N,)  model's predicted RUL at each test point.

All functions return floats; use `score_bundle()` at the bottom to get
the full scorecard as a flat dict for MLflow logging.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArr = npt.NDArray[np.floating]


# ─── Core metrics ─────────────────────────────────────────────────────


def nasa_s_score(
    y_true_rul: FloatArr,
    y_pred_rul: FloatArr,
    early_scale: float = 13.0,
    late_scale: float = 10.0,
) -> float:
    """NASA S-score — the canonical CMAPSS prognostics metric.

    Asymmetric exponential penalty: late predictions (over-estimating
    RUL) cost more than early ones (under-estimating), because in
    prognostics a false-negative-on-failure has operational consequences
    that a premature maintenance flag does not.

    S(error) = exp(error / late_scale) - 1   for late  predictions (error > 0)
    S(error) = exp(-error / early_scale) - 1 for early predictions (error < 0)

    where `error = y_pred - y_true`. Total score is the sum (not mean)
    of per-sample S values, mirroring the original NASA formulation.

    Lower is better; 0 = perfect prediction.
    """
    err = y_pred_rul - y_true_rul
    late = err > 0
    early = ~late
    s = np.empty_like(err, dtype=np.float64)
    s[late] = np.exp(err[late] / late_scale) - 1.0
    s[early] = np.exp(-err[early] / early_scale) - 1.0
    return float(np.sum(s))


def rul_mae(y_true_rul: FloatArr, y_pred_rul: FloatArr) -> float:
    """Mean absolute error on RUL predictions."""
    return float(np.mean(np.abs(y_true_rul - y_pred_rul)))


def rul_rmse(y_true_rul: FloatArr, y_pred_rul: FloatArr) -> float:
    """Root mean squared error on RUL predictions."""
    return float(np.sqrt(np.mean((y_true_rul - y_pred_rul) ** 2)))


def hit_rate_at_tolerance(
    y_true_rul: FloatArr,
    y_pred_rul: FloatArr,
    tolerance: float,
) -> float:
    """Fraction of predictions within +/- `tolerance` units of the truth.

    "Hit rate at 7 days" = fraction of RUL predictions that are within
    a week of the true remaining life. Complements the mean-error
    metrics with a tail-behavior view.
    """
    return float(np.mean(np.abs(y_true_rul - y_pred_rul) <= tolerance))


def early_late_split(
    y_true_rul: FloatArr,
    y_pred_rul: FloatArr,
) -> dict[str, float]:
    """Split MAE by direction of error for diagnostic purposes.

    Returns:
        early_mae: MAE over samples where the model under-predicted
                   (safe direction — extra maintenance).
        late_mae:  MAE over samples where the model over-predicted
                   (unsafe direction — missed failure warning).
        late_fraction: fraction of predictions that were late.

    A well-behaved RUL model should have `late_mae` close to `early_mae`
    (unbiased) OR should skew toward early errors (conservative bias).
    A large `late_mae` with a high `late_fraction` is the failure mode
    the NASA S-score is designed to punish.
    """
    err = y_pred_rul - y_true_rul
    late = err > 0
    early = ~late
    late_count = int(late.sum())
    early_count = int(early.sum())
    late_mae = float(np.mean(np.abs(err[late]))) if late_count > 0 else 0.0
    early_mae = float(np.mean(np.abs(err[early]))) if early_count > 0 else 0.0
    return {
        "early_mae": early_mae,
        "late_mae": late_mae,
        "late_fraction": float(late_count / (late_count + early_count)) if (late_count + early_count) > 0 else 0.0,
    }


# ─── Calibration data (for plots, not a scalar metric) ────────────────


def calibration_pairs(
    y_true_rul: FloatArr,
    y_pred_rul: FloatArr,
    n_bins: int = 20,
) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Return (bin_centers, mean_pred_in_bin, mean_true_in_bin) for a
    calibration scatter plot.

    Bins by true RUL. A perfectly calibrated model has
    `mean_pred_in_bin == bin_centers` (points on the y=x diagonal).
    """
    lo, hi = float(y_true_rul.min()), float(y_true_rul.max())
    if hi <= lo:
        # Degenerate case: all truths identical.
        return np.array([lo]), np.array([float(np.mean(y_pred_rul))]), np.array([lo])
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_pred = np.empty(n_bins)
    mean_true = np.empty(n_bins)
    for i in range(n_bins):
        mask = (y_true_rul >= edges[i]) & (y_true_rul < edges[i + 1])
        if mask.sum() == 0:
            mean_pred[i] = np.nan
            mean_true[i] = np.nan
        else:
            mean_pred[i] = float(y_pred_rul[mask].mean())
            mean_true[i] = float(y_true_rul[mask].mean())
    return centers, mean_pred, mean_true


# ─── Convenience: full scorecard ──────────────────────────────────────


def score_bundle(
    y_true_rul: FloatArr,
    y_pred_rul: FloatArr,
    tolerances: tuple[float, ...] = (1.0, 7.0, 30.0),
) -> dict[str, float]:
    """Full RUL scorecard as a flat dict, ready for MLflow.

    `tolerances` are the "hit rate at N units" evaluation points; defaults
    correspond to 1 day / 1 week / 1 month for daily-aggregated RUL.
    """
    out: dict[str, float] = {
        "nasa_s_score": nasa_s_score(y_true_rul, y_pred_rul),
        "mae": rul_mae(y_true_rul, y_pred_rul),
        "rmse": rul_rmse(y_true_rul, y_pred_rul),
    }
    out.update(early_late_split(y_true_rul, y_pred_rul))
    for tol in tolerances:
        # Encode tolerance in the key so multiple horizons don't collide.
        out[f"hit_rate_tol_{tol:g}"] = hit_rate_at_tolerance(
            y_true_rul, y_pred_rul, tolerance=tol
        )
    return out
