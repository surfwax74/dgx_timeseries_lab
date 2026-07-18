"""Forecasting metrics — point + probabilistic.

Standard time-series-competition scorers. Every function accepts numpy
arrays with a consistent shape convention:

    y_true:      (N, H)    N test windows, H forecast horizon steps
    y_pred:      (N, H)    point predictions
    y_quantiles: (N, H, Q) quantile predictions; caller supplies the Q levels

Returned scalars are averaged over both N and H unless the docstring
says otherwise.

Design decisions
----------------
* MASE uses a naive seasonal forecast baseline (persistence with
  seasonal period `season_length`, default 1 = last observation).
* sMAPE uses the "symmetric" formulation with `2*|y-yhat| / (|y|+|yhat|)`;
  the M4-competition formulation is identical.
* Pinball loss follows the standard `max(q*(y-qhat), (q-1)*(y-qhat))`.
* CRPS is computed empirically from the quantile grid via trapezoidal
  integration — good enough for our M-quantile grids (usually M <= 11).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArr = npt.NDArray[np.floating]


# ─── Point-forecast metrics ───────────────────────────────────────────


def mae(y_true: FloatArr, y_pred: FloatArr) -> float:
    """Mean absolute error over all (N, H) entries."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: FloatArr, y_pred: FloatArr) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: FloatArr, y_pred: FloatArr, eps: float = 1e-8) -> float:
    """Symmetric MAPE, M4-competition formulation. Bounded [0, 200]."""
    num = 2.0 * np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(num / den) * 100.0)


def mase(
    y_true: FloatArr,
    y_pred: FloatArr,
    y_train: FloatArr,
    season_length: int = 1,
) -> float:
    """Mean absolute scaled error.

    Scale is the in-sample MAE of a naive seasonal forecast on `y_train`
    (persistence with lag `season_length`). MASE < 1 beats the naive
    baseline; MASE > 1 loses to it. `y_train` should be the training
    portion of the same series — long enough that the scale estimate is
    stable (>= 2 * season_length observations).
    """
    if y_train.ndim > 1:
        y_train = y_train.ravel()
    if len(y_train) <= season_length:
        raise ValueError(
            f"MASE requires len(y_train) > season_length "
            f"(got {len(y_train)} <= {season_length})"
        )
    naive_error = np.abs(y_train[season_length:] - y_train[:-season_length])
    scale = float(np.mean(naive_error))
    if scale < 1e-12:
        # Perfectly flat training signal — MASE is undefined; return raw MAE
        # instead of dividing by ~0.
        return mae(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


# ─── Probabilistic metrics ────────────────────────────────────────────


def pinball_loss(y_true: FloatArr, y_quantile: FloatArr, q: float) -> float:
    """Quantile (pinball) loss for a single quantile level q in (0, 1)."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1); got {q}")
    diff = y_true - y_quantile
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def crps_from_quantiles(
    y_true: FloatArr,
    y_quantiles: FloatArr,
    quantile_levels: FloatArr | list[float],
) -> float:
    """Empirical CRPS via trapezoidal integration over the quantile grid.

    Under the standard identity `CRPS = 2 * integral over q of pinball(q) dq`,
    integrating with the trapezoidal rule over the supplied quantile levels
    gives a decent estimate for grids >= 9 quantiles. For a fine grid
    (0.01, 0.02, ..., 0.99) it converges to the analytic CRPS.

    Args:
        y_true: (N, H) actuals.
        y_quantiles: (N, H, Q) predicted quantiles. Must be sorted along Q
            in the same order as `quantile_levels`.
        quantile_levels: length-Q array of quantile levels in (0, 1),
            monotonically increasing.
    """
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if levels.ndim != 1:
        raise ValueError("quantile_levels must be 1D")
    if not np.all(np.diff(levels) > 0):
        raise ValueError("quantile_levels must be strictly increasing")
    if y_quantiles.shape[-1] != len(levels):
        raise ValueError(
            f"y_quantiles last dim ({y_quantiles.shape[-1]}) "
            f"!= len(quantile_levels) ({len(levels)})"
        )

    per_quantile_loss = np.stack(
        [
            2.0 * np.maximum(q * (y_true - y_quantiles[..., i]),
                             (q - 1.0) * (y_true - y_quantiles[..., i]))
            for i, q in enumerate(levels)
        ],
        axis=-1,
    )
    # Trapezoidal integration over the quantile axis.
    integrated = np.trapezoid(per_quantile_loss, x=levels, axis=-1)
    return float(np.mean(integrated))


def interval_coverage(
    y_true: FloatArr,
    lower: FloatArr,
    upper: FloatArr,
) -> float:
    """Fraction of `y_true` inside the [lower, upper] band. Calibration check.

    A well-calibrated 80% interval should give coverage ~0.80; a 95%
    interval ~0.95. Systematic under-coverage means the model is
    over-confident.
    """
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


# ─── Convenience: full metric bundle ──────────────────────────────────


def score_bundle(
    y_true: FloatArr,
    y_pred: FloatArr,
    y_train: FloatArr | None = None,
    season_length: int = 1,
    y_quantiles: FloatArr | None = None,
    quantile_levels: FloatArr | list[float] | None = None,
) -> dict[str, float]:
    """Compute the full forecasting scorecard in one call.

    Returns a flat dict suitable for MLflow logging. Probabilistic
    metrics are omitted if `y_quantiles` is None; MASE is omitted if
    `y_train` is None.
    """
    out: dict[str, float] = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
    if y_train is not None:
        out["mase"] = mase(y_true, y_pred, y_train, season_length=season_length)
    if y_quantiles is not None and quantile_levels is not None:
        levels = np.asarray(quantile_levels, dtype=np.float64)
        out["crps"] = crps_from_quantiles(y_true, y_quantiles, levels)
        for q in (0.1, 0.5, 0.9):
            if q in levels:
                idx = int(np.where(levels == q)[0][0])
                out[f"pinball_q{int(q * 100):02d}"] = pinball_loss(
                    y_true, y_quantiles[..., idx], q
                )
        if 0.1 in levels and 0.9 in levels:
            lo = y_quantiles[..., int(np.where(levels == 0.1)[0][0])]
            hi = y_quantiles[..., int(np.where(levels == 0.9)[0][0])]
            out["coverage_80"] = interval_coverage(y_true, lo, hi)
    return out
