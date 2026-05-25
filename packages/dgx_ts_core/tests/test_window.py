"""Unit tests for TelemetryWindow."""

from __future__ import annotations

import numpy as np
import pytest

from dgx_ts_core.data import Channel, Subsystem, TelemetryWindow, Units


def _ch(n: int) -> tuple[Channel, ...]:
    return tuple(
        Channel(name=f"ch{i}", units=Units.DIMENSIONLESS, subsystem=Subsystem.UNKNOWN, sample_rate_hz=1.0)
        for i in range(n)
    )


def test_window_basic_construction() -> None:
    t, c = 32, 3
    w = TelemetryWindow(
        tensor=np.zeros((t, c), dtype=np.float32),
        timestamps=np.arange(t, dtype=np.int64),
        channels=_ch(c),
    )
    assert w.length == t
    assert w.num_channels == c
    assert w.mask is None
    assert w.labels is None


def test_window_with_mask_and_labels() -> None:
    t, c = 8, 2
    w = TelemetryWindow(
        tensor=np.ones((t, c), dtype=np.float32),
        timestamps=np.arange(t, dtype=np.int64),
        channels=_ch(c),
        mask=np.zeros((t, c), dtype=np.bool_),
        labels=np.array([False] * 7 + [True], dtype=np.bool_),
    )
    assert w.labels is not None and w.labels[-1]
    assert w.mask is not None and not w.mask.any()


def test_window_rejects_wrong_tensor_rank() -> None:
    with pytest.raises(ValueError, match="2-D"):
        TelemetryWindow(
            tensor=np.zeros(10, dtype=np.float32),  # 1-D
            timestamps=np.arange(10, dtype=np.int64),
            channels=_ch(1),
        )


def test_window_rejects_mismatched_channel_count() -> None:
    with pytest.raises(ValueError, match="channel count"):
        TelemetryWindow(
            tensor=np.zeros((4, 3), dtype=np.float32),
            timestamps=np.arange(4, dtype=np.int64),
            channels=_ch(2),
        )


def test_window_rejects_mismatched_label_length() -> None:
    with pytest.raises(ValueError, match="labels shape"):
        TelemetryWindow(
            tensor=np.zeros((4, 2), dtype=np.float32),
            timestamps=np.arange(4, dtype=np.int64),
            channels=_ch(2),
            labels=np.zeros(3, dtype=np.bool_),
        )


def test_window_is_immutable() -> None:
    t, c = 4, 2
    w = TelemetryWindow(
        tensor=np.zeros((t, c), dtype=np.float32),
        timestamps=np.arange(t, dtype=np.int64),
        channels=_ch(c),
    )
    with pytest.raises((AttributeError, TypeError)):
        w.tensor = np.ones((t, c), dtype=np.float32)  # type: ignore[misc]
