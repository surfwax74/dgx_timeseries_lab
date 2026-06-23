"""Resample bursty discrete events (commands, logs) onto a fixed-Hz time grid.

Locked Phase 10 encoding: per bin we emit a 3-feature vector summarizing
what happened in that bin. Zero-padded when no events.

    Commands bin:  [count, last_opcode_id, last_param_id]
    Logs bin:      [count, max_severity_int, last_source_id]

Output is always shape ``(n_bins, 3)`` float32 — slots into the standard
TelemetryDataset tensor format.
"""

from __future__ import annotations

import numpy as np


class CommandEventBucketer:
    """Resample timestamped command events onto a 1-Hz (or any) grid."""

    n_features: int = 3

    def __init__(self, sample_rate_hz: float = 1.0) -> None:
        self.sample_rate_hz = float(sample_rate_hz)

    def bucket(
        self,
        event_times_s: np.ndarray,        # (N,) event times in seconds
        opcode_ids: np.ndarray,           # (N,) int
        param_ids: np.ndarray,            # (N,) int (first param of each cmd)
        n_bins: int,
    ) -> np.ndarray:
        """Returns (n_bins, 3) float32: [count, last_opcode_id, last_param_id]."""
        bin_width_s = 1.0 / self.sample_rate_hz
        out = np.zeros((n_bins, 3), dtype=np.float32)
        if event_times_s.size == 0:
            return out
        bin_idx = np.floor(event_times_s / bin_width_s).astype(np.int64)
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx = bin_idx[valid]
        opc = opcode_ids[valid]
        prm = param_ids[valid]
        # Count per bin
        np.add.at(out[:, 0], bin_idx, 1.0)
        # Last opcode + last param per bin: iterate in order so later overwrites earlier
        for b, op_, p in zip(bin_idx, opc, prm, strict=False):
            out[b, 1] = float(op_)
            out[b, 2] = float(p)
        return out


class LogEventBucketer:
    """Resample timestamped log events onto a fixed-Hz grid."""

    n_features: int = 3

    def __init__(self, sample_rate_hz: float = 1.0) -> None:
        self.sample_rate_hz = float(sample_rate_hz)

    def bucket(
        self,
        event_times_s: np.ndarray,        # (N,)
        severity_ints: np.ndarray,        # (N,) int (LogSeverity values)
        source_ids: np.ndarray,           # (N,) int
        n_bins: int,
    ) -> np.ndarray:
        """Returns (n_bins, 3) float32: [count, max_severity, last_source_id]."""
        bin_width_s = 1.0 / self.sample_rate_hz
        out = np.zeros((n_bins, 3), dtype=np.float32)
        if event_times_s.size == 0:
            return out
        bin_idx = np.floor(event_times_s / bin_width_s).astype(np.int64)
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx = bin_idx[valid]
        sev = severity_ints[valid]
        src = source_ids[valid]
        np.add.at(out[:, 0], bin_idx, 1.0)
        # Max severity + last source per bin
        for b, s, src_ in zip(bin_idx, sev, src, strict=False):
            if s > out[b, 1]:
                out[b, 1] = float(s)
            out[b, 2] = float(src_)
        return out
