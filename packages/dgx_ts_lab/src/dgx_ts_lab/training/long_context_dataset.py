"""Chunked sliding-window dataset for very long sequences.

Standard `WindowTorchDataset` materializes window starts eagerly. For
multi-day / 10⁵+ timestep datasets, that index list gets big and you
want streaming + pre-fetch rather than random shuffle.

This dataset trades shuffle quality for memory efficiency: it shuffles
WITHIN chunks of ``shuffle_buffer_size`` consecutive starts, not across
the whole dataset. Good enough for AD pre-training where temporal locality
within a chunk is not catastrophic.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from dgx_ts_core.data import TelemetryDataset


class LongContextTorchDataset(IterableDataset):
    """Streaming windowed view of a TelemetryDataset.

    Use for window lengths >= 1024 or dataset sizes >= 10⁵ timesteps where
    materializing every window start in a list becomes expensive.

    Workers each see a disjoint stride of the dataset, with shuffle inside
    each worker's buffer.
    """

    def __init__(
        self,
        dataset: TelemetryDataset,
        length: int,
        stride: int,
        shuffle_buffer_size: int = 1024,
        seed: int = 0,
    ) -> None:
        if not hasattr(dataset, "_data"):
            raise TypeError(
                f"{type(dataset).__name__} doesn't expose _data; "
                "long-context streaming needs the dense array. "
                "Use WindowTorchDataset for generator-only datasets."
            )
        self._data: np.ndarray = dataset._data
        self._labels: np.ndarray | None = getattr(dataset, "_labels", None)
        self._length = int(length)
        self._stride = int(stride)
        self._shuffle_buf = int(shuffle_buffer_size)
        self._seed = int(seed)

    def __iter__(self):
        info = get_worker_info()
        n_workers = 1 if info is None else info.num_workers
        worker_id = 0 if info is None else info.id

        n = int(self._data.shape[0])
        if self._length > n:
            return
        all_starts = list(range(0, n - self._length + 1, self._stride))
        # Each worker takes every Nth start, with offset = worker_id
        my_starts = all_starts[worker_id::n_workers]
        rng = np.random.default_rng(self._seed + worker_id)

        # Shuffle inside the buffer for some randomness
        for chunk_start in range(0, len(my_starts), self._shuffle_buf):
            chunk = my_starts[chunk_start : chunk_start + self._shuffle_buf]
            rng.shuffle(chunk)
            for start in chunk:
                end = start + self._length
                x = torch.from_numpy(self._data[start:end].copy()).float()
                if self._labels is not None:
                    labels = torch.from_numpy(self._labels[start:end].copy()).bool()
                else:
                    labels = torch.zeros(self._length, dtype=torch.bool)
                yield {"x": x, "labels": labels}
