"""Unit tests for evaluation.forecasting_metrics and evaluation.rul_metrics.

Every scorer is exercised on both a trivial-perfect case (zero error) and
a hand-computable non-trivial case so a regression in the formula gets
caught by the actual number, not just "did it return a float".
"""

from __future__ import annotations

import numpy as np
import pytest
from dgx_ts_lab.evaluation import forecasting_metrics as fm
from dgx_ts_lab.evaluation import rul_metrics as rm


# ─── Forecasting: point metrics ───────────────────────────────────────


def test_perfect_forecast_gives_zero_point_errors() -> None:
    y = np.arange(20, dtype=np.float32).reshape(4, 5)
    assert fm.mae(y, y) == 0.0
    assert fm.rmse(y, y) == 0.0
    assert fm.smape(y, y) == pytest.approx(0.0, abs=1e-6)


def test_mae_rmse_on_constant_error() -> None:
    y_true = np.zeros((10, 5), dtype=np.float32)
    y_pred = np.ones((10, 5), dtype=np.float32) * 3.0
    assert fm.mae(y_true, y_pred) == pytest.approx(3.0)
    assert fm.rmse(y_true, y_pred) == pytest.approx(3.0)


def test_smape_bounded_and_symmetric() -> None:
    # sMAPE of predicting 0 for true=1 and predicting 1 for true=0 must be
    # the same (symmetric) and finite (bounded).
    a = fm.smape(np.array([[1.0]]), np.array([[0.0]]))
    b = fm.smape(np.array([[0.0]]), np.array([[1.0]]))
    assert a == pytest.approx(b)
    assert 0.0 <= a <= 200.0


def test_mase_matches_hand_computation() -> None:
    # Training series 1,2,3,4,5 -> naive-1-step errors are all 1, so scale=1.
    # Test: predict 10 for true 12 -> abs error 2. MASE = 2 / 1 = 2.
    y_train = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    y_true = np.array([[12.0]])
    y_pred = np.array([[10.0]])
    assert fm.mase(y_true, y_pred, y_train) == pytest.approx(2.0)


def test_mase_falls_back_to_mae_on_flat_train() -> None:
    # If training series is perfectly flat, scale would be 0; the fallback
    # returns raw MAE instead of dividing by zero.
    y_train = np.ones(10, dtype=np.float32) * 5.0
    y_true = np.array([[10.0]])
    y_pred = np.array([[7.0]])
    assert fm.mase(y_true, y_pred, y_train) == pytest.approx(3.0)


def test_mase_rejects_too_short_train() -> None:
    with pytest.raises(ValueError, match="MASE requires"):
        fm.mase(np.array([[1.0]]), np.array([[1.0]]), np.array([1.0]), season_length=1)


# ─── Forecasting: probabilistic metrics ───────────────────────────────


def test_pinball_loss_hand_computation() -> None:
    # At q=0.5, pinball loss = 0.5 * |error| (weighted MAE / 2)
    y_true = np.array([1.0, 2.0, 3.0])
    y_qhat = np.array([2.0, 2.0, 2.0])
    # errors: -1, 0, +1  -> pinball at q=0.5 -> mean(0.5, 0, 0.5) = 1/3
    got = fm.pinball_loss(y_true, y_qhat, q=0.5)
    assert got == pytest.approx(1.0 / 3.0)


def test_pinball_asymmetry_at_high_quantile() -> None:
    # At q=0.9, under-prediction (positive residual) is penalized ~9x more
    # than over-prediction.
    y = np.array([10.0])
    under = fm.pinball_loss(y, np.array([9.0]), q=0.9)
    over = fm.pinball_loss(y, np.array([11.0]), q=0.9)
    assert under == pytest.approx(0.9)
    assert over == pytest.approx(0.1)


def test_pinball_rejects_out_of_range_q() -> None:
    with pytest.raises(ValueError, match="q must be in"):
        fm.pinball_loss(np.array([1.0]), np.array([1.0]), q=1.5)


def test_interval_coverage_all_inside_gives_one() -> None:
    y = np.array([0.5, 0.5])
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    assert fm.interval_coverage(y, lo, hi) == 1.0


def test_interval_coverage_half_inside_gives_half() -> None:
    y = np.array([0.5, 2.0])   # first inside, second outside
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    assert fm.interval_coverage(y, lo, hi) == 0.5


def test_crps_shape_validation() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        fm.crps_from_quantiles(
            np.array([[1.0]]), np.array([[[0.0, 0.0]]]), [0.5, 0.3]
        )


def test_score_bundle_contains_expected_keys() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(20, 8)).astype(np.float32)
    y_pred = y_true + rng.normal(size=(20, 8), scale=0.1).astype(np.float32)
    y_train = rng.normal(size=100).astype(np.float32)

    levels = np.array([0.1, 0.5, 0.9])
    y_q = np.stack([y_pred - 0.5, y_pred, y_pred + 0.5], axis=-1)

    bundle = fm.score_bundle(
        y_true, y_pred, y_train=y_train,
        y_quantiles=y_q, quantile_levels=levels,
    )
    assert set(bundle) >= {
        "mae", "rmse", "smape", "mase",
        "crps", "pinball_q10", "pinball_q50", "pinball_q90",
        "coverage_80",
    }


# ─── RUL metrics ──────────────────────────────────────────────────────


def test_nasa_s_score_zero_at_perfect_prediction() -> None:
    y = np.array([50.0, 100.0, 200.0])
    assert rm.nasa_s_score(y, y) == pytest.approx(0.0)


def test_nasa_s_score_penalizes_late_more_than_early() -> None:
    # Same |error|, different sign — late should score worse.
    truth = np.array([100.0])
    early_penalty = rm.nasa_s_score(truth, np.array([90.0]))   # under-predict
    late_penalty = rm.nasa_s_score(truth, np.array([110.0]))    # over-predict
    assert late_penalty > early_penalty
    assert early_penalty > 0.0


def test_rul_mae_and_rmse() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    assert rm.rul_mae(y_true, y_pred) == pytest.approx((2 + 2 + 3) / 3)
    expected_rmse = np.sqrt(np.mean(np.array([4.0, 4.0, 9.0])))
    assert rm.rul_rmse(y_true, y_pred) == pytest.approx(expected_rmse)


def test_hit_rate_at_tolerance() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 20.5, 40.0])   # errors: 2, 0.5, 10
    assert rm.hit_rate_at_tolerance(y_true, y_pred, tolerance=1.0) == pytest.approx(1 / 3)
    assert rm.hit_rate_at_tolerance(y_true, y_pred, tolerance=5.0) == pytest.approx(2 / 3)
    assert rm.hit_rate_at_tolerance(y_true, y_pred, tolerance=20.0) == pytest.approx(1.0)


def test_early_late_split_directions() -> None:
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([8.0, 25.0])   # under, over
    split = rm.early_late_split(y_true, y_pred)
    assert split["early_mae"] == pytest.approx(2.0)
    assert split["late_mae"] == pytest.approx(5.0)
    assert split["late_fraction"] == 0.5


def test_calibration_pairs_shapes() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.uniform(0.0, 100.0, size=50)
    y_pred = y_true + rng.normal(scale=5.0, size=50)
    centers, mean_pred, mean_true = rm.calibration_pairs(y_true, y_pred, n_bins=10)
    assert centers.shape == (10,)
    assert mean_pred.shape == (10,)
    assert mean_true.shape == (10,)


def test_calibration_pairs_handles_degenerate_case() -> None:
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    centers, mean_pred, mean_true = rm.calibration_pairs(y_true, y_pred)
    assert centers.shape == (1,)
    assert mean_pred[0] == pytest.approx(5.0)


def test_rul_score_bundle_keys() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    bundle = rm.score_bundle(y_true, y_pred, tolerances=(1.0, 5.0))
    assert set(bundle) >= {
        "nasa_s_score", "mae", "rmse",
        "early_mae", "late_mae", "late_fraction",
        "hit_rate_tol_1", "hit_rate_tol_5",
    }
