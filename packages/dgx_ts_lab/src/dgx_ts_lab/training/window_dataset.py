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
            return {"x": x, "labels": labels}
        # Materialized fallback
        assert self._materialized is not None
        w = self._materialized[idx]
        x = torch.from_numpy(w.tensor.copy()).float()
        labels = (
            torch.from_numpy(w.labels.copy()).bool()
            if w.labels is not None
            else torch.zeros(w.length, dtype=torch.bool)
        )
        return {"x": x, "labels": labels}
