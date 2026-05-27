"""Torch Dataset wrapping a `TelemetryDataset` for use with `DataLoader`.

Random-access (so shuffle works) via the underlying dense arrays — every
bundled dataset exposes ``_data``, ``_labels``, ``_timestamps``. If you
implement a TelemetryDataset that doesn't, the materialized fallback
walks ``windows()`` once at construction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from dgx_ts_core.data import TelemetryDataset


class WindowTorchDataset(Dataset):
    """Indexable view of a TelemetryDataset's sliding windows."""

    def __init__(
        self,
        dataset: TelemetryDataset,
        length: int,
        stride: int,
    ) -> None:
        self._length = int(length)
        self._stride = int(stride)
        self._name = dataset.name
        # Stash for Phase 6 aux_labels passthrough on the fast path.
        self._dataset = dataset

        # Fast path: underlying dense arrays
        if hasattr(dataset, "_data"):
            self._data: np.ndarray | None = dataset._data
            self._labels: np.ndarray | None = getattr(dataset, "_labels", None)
            n = int(self._data.shape[0])
            if self._length > n:
                self._starts: list[int] = []
            else:
                self._starts = list(range(0, n - self._length + 1, self._stride))
            self._materialized: list | None = None
        else:
            # Generic fallback: materialize all windows once
            self._data = None
            self._labels = None
            self._materialized = list(dataset.windows(length=length, stride=stride))
            self._starts = list(range(len(self._materialized)))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._data is not None:
            start = self._starts[idx]
            end = start + self._length
            x = torch.from_numpy(self._data[start:end].copy()).float()  # (T, C)
            if self._labels is not None:
                labels = torch.from_numpy(self._labels[start:end].copy()).bool()
            else:
                labels = torch.zeros(self._length, dtype=torch.bool)
            out: dict[str, torch.Tensor] = {"x": x, "labels": labels}
            aux = self._aux_labels_for(start, self._length)
            if aux is not None:
                out["aux_labels"] = aux
            return out
        # Materialized fallback
        assert self._materialized is not None
        w = self._materialized[idx]
        x = torch.from_numpy(w.tensor.copy()).float()
        labels = (
            torch.from_numpy(w.labels.copy()).bool()
            if w.labels is not None
            else torch.zeros(w.length, dtype=torch.bool)
        )
        out = {"x": x, "labels": labels}
        if w.aux_labels is not None:
            out["aux_labels"] = {
                k: torch.from_numpy(v.copy()) for k, v in w.aux_labels.items()
            }
        return out

    def _aux_labels_for(self, start: int, length: int) -> dict[str, torch.Tensor] | None:
        """Phase 6: cached per-window aux-label fetch from the underlying dataset."""
        ds = getattr(self, "_dataset", None)
        if ds is None:
            return None
        if hasattr(ds, "_ensure_label_computer"):
            ds._ensure_label_computer()
            lc = getattr(ds, "_label_computer", None)
            if lc is not None:
                aux_np = lc.labels_for_window(start, length)
                return {k: torch.from_numpy(v) for k, v in aux_np.items()}
        return None
