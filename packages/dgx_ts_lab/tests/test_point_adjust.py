"""Tests for point-adjusted F1."""

from __future__ import annotations

import numpy as np
from dgx_ts_lab.evaluation import point_adjust, point_adjusted_metrics


def test_point_adjust_expands_to_full_segment_when_any_hit() -> None:
    y_true = np.array([False, False, True, True, True, False], dtype=np.bool_)
    y_pred = np.array([False, False, False, True, False, False], dtype=np.bool_)
    adj = point_adjust(y_true, y_pred)
    # The true segment [2, 5) had one hit (idx 3) → all of [2, 5) becomes True
    assert adj.tolist() == [False, False, True, True, True, False]


def test_point_adjust_leaves_segment_alone_when_no_hit() -> None:
    y_true = np.array([False, True, True, False, False], dtype=np.bool_)
    y_pred = np.array([False, False, False, True, False], dtype=np.bool_)
    adj = point_adjust(y_true, y_pred)
    # The true segment [1, 3) had no hit → stays False
    assert adj.tolist() == [False, False, False, True, False]


def test_point_adjust_handles_empty_labels() -> None:
    y_true = np.zeros(5, dtype=np.bool_)
    y_pred = np.array([False, True, False, True, False], dtype=np.bool_)
    adj = point_adjust(y_true, y_pred)
    assert (adj == y_pred).all()


def test_point_adjusted_f1_higher_than_raw_for_segment_with_hit() -> None:
    from sklearn.metrics import f1_score

    y_true = np.zeros(20, dtype=np.bool_)
    y_true[5:15] = True  # one 10-step segment
    scores = np.zeros(20, dtype=np.float32)
    scores[7] = 5.0  # one hit inside segment
    threshold = 1.0

    raw_f1 = f1_score(y_true, scores > threshold, zero_division=0)
    pa = point_adjusted_metrics(y_true, scores, threshold=threshold)
    assert pa["pa_f1"] > raw_f1
    assert pa["pa_recall"] == 1.0
