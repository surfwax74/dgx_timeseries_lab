"""Multi-task label generator for the layered synthetic dataset.

Derives three per-step label arrays from the dataset's existing
``fault_log`` + ``mode_trace``:

    fault_type:  (T,) int64 — class index (0 == no fault, 1..K == fault types)
    rul:         (T,) float32 — seconds until next fault (or NO_FAULT_CEILING_SECONDS)
    next_mode:   (T,) int64 — mode at t + horizon_steps (or -1 sentinel if out of range)

These are consumed by Phase 6's task heads via
``TelemetryWindow.aux_labels``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Class index 0 reserved for "no fault active".
# Mapping is alphabetical over the known fault `kind` strings produced by
# layered.faults so the index is deterministic.
FAULT_CLASS_INDEX: dict[str, int] = {
    "no_fault": 0,
    "correlation_break_fault": 1,
    "drift_fault": 2,
    "dropout_fault": 3,
    "mode_confusion_fault": 4,
    "oscillation_fault": 5,
    "point_fault": 6,
    "stuck_at_fault": 7,
}
NUM_FAULT_CLASSES = len(FAULT_CLASS_INDEX)

# RUL ceiling for "no upcoming fault" — flagged so the head can mask the loss.
NO_FAULT_CEILING_SECONDS = 1e9


class MultiTaskLabelComputer:
    """Pre-computes full-length label arrays from fault_log + mode_trace.

    Construction is O(N + T) where N is the number of fault events and T
    is the dataset length. Per-window slicing is O(window_length).
    """

    def __init__(
        self,
        fault_log: list[dict[str, Any]],
        mode_trace: np.ndarray,
        sample_rate_hz: float,
        next_mode_horizon_s: float = 60.0,
    ) -> None:
        self._n = int(mode_trace.shape[0])
        self._sr = float(sample_rate_hz)
        self._horizon_steps = max(1, int(next_mode_horizon_s * self._sr))
        self._fault_type = self._build_fault_type_array(fault_log)
        self._rul = self._build_rul_array(fault_log)
        self._next_mode = self._build_next_mode_array(mode_trace)

    # ── builders ─────────────────────────────────────────────────────────

    def _build_fault_type_array(
        self, fault_log: list[dict[str, Any]]
    ) -> np.ndarray:
        out = np.zeros(self._n, dtype=np.int64)  # 0 == no_fault
        for entry in fault_log:
            kind = entry.get("type", "no_fault")
            cls = FAULT_CLASS_INDEX.get(kind, 0)
            if cls == 0:
                continue
            s = int(entry.get("start", 0))
            e = int(entry.get("end", s + 1))
            if 0 <= s < self._n:
                out[s:min(e, self._n)] = cls
        return out

    def _build_rul_array(self, fault_log: list[dict[str, Any]]) -> np.ndarray:
        """For each step t, time (seconds) until the NEXT fault onset
        anywhere in the dataset (across all channels)."""
        out = np.full(self._n, NO_FAULT_CEILING_SECONDS, dtype=np.float32)
        if not fault_log:
            return out
        # Sort fault onsets ascending; walk backwards filling RUL.
        starts = sorted(int(e.get("start", 0)) for e in fault_log)
        # For each onset, fill out[t] for all t in [prev_onset_or_0, this_onset)
        # with this_onset - t (in seconds).
        prev = 0
        for s in starts:
            s_cl = min(max(s, 0), self._n)
            if s_cl <= prev:
                continue
            indices = np.arange(prev, s_cl, dtype=np.float32)
            out[prev:s_cl] = (s_cl - indices) / self._sr
            prev = s_cl
        # Everything after the last onset keeps the ceiling.
        return out

    def _build_next_mode_array(self, mode_trace: np.ndarray) -> np.ndarray:
        """Mode at (t + horizon_steps), with -1 sentinel where out of range."""
        out = np.full(self._n, -1, dtype=np.int64)
        h = self._horizon_steps
        if h < self._n:
            out[: self._n - h] = mode_trace[h:].astype(np.int64)
        return out

    # ── per-window slicing ──────────────────────────────────────────────

    def labels_for_window(self, start: int, length: int) -> dict[str, np.ndarray]:
        end = min(start + length, self._n)
        actual = end - start
        ft = self._fault_type[start:end].copy()
        ru = self._rul[start:end].copy()
        nm = self._next_mode[start:end].copy()
        if actual < length:
            ft = np.concatenate([ft, np.zeros(length - actual, dtype=np.int64)])
            ru = np.concatenate([ru, np.full(length - actual, NO_FAULT_CEILING_SECONDS, dtype=np.float32)])
            nm = np.concatenate([nm, np.full(length - actual, -1, dtype=np.int64)])
        return {"fault_type": ft, "rul": ru, "next_mode": nm}

    @property
    def fault_type_full(self) -> np.ndarray:
        return self._fault_type

    @property
    def rul_full(self) -> np.ndarray:
        return self._rul

    @property
    def next_mode_full(self) -> np.ndarray:
        return self._next_mode
