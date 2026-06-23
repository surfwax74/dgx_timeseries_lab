"""CommandSequenceDataset — discrete command token stream as TelemetryDataset.

Each "window" is ``length`` consecutive tokens. tensor shape is ``(length, 1)``
float32 (token IDs cast from int — the sequence detector casts back to long
at forward time).

Anomaly labels propagate from the underlying token-level labels (True for
tokens injected as anomalies).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from dgx_ts_core.data import (
    Channel,
    DatasetStats,
    SplitScheme,
    Subsystem,
    TelemetryWindow,
    Units,
)
from dgx_ts_core.registry import DATASET_REGISTRY

from ._tokenizer import CommandTokenizer


class CommandSequenceDataset:
    """A TelemetryDataset over a flat command-token stream."""

    def __init__(
        self,
        tokens: np.ndarray,                  # (N,) int64
        labels: np.ndarray | None = None,    # (N,) bool — True = injection
        tokenizer: CommandTokenizer | None = None,
        # Optional: per-token aux labels (e.g., "injection_type": int)
        aux: dict[str, np.ndarray] | None = None,
        sample_rate_hz: float = 1.0,
        name: str = "command_sequence",
        subsystem: Subsystem = Subsystem.OBDH,
        _materialized: dict | None = None,
    ) -> None:
        if _materialized is not None:
            self._tokens = _materialized["tokens"]
            self._labels = _materialized["labels"]
            self._aux = _materialized.get("aux")
            self._tokenizer = _materialized.get("tokenizer")
            self._sample_rate_hz = float(_materialized["sample_rate_hz"])
            self._name = _materialized["name"]
            self._subsystem = _materialized["subsystem"]
        else:
            self._tokens = tokens.astype(np.int64)
            self._labels = (
                labels.astype(np.bool_) if labels is not None
                else np.zeros(len(self._tokens), dtype=np.bool_)
            )
            self._aux = aux
            self._tokenizer = tokenizer
            self._sample_rate_hz = float(sample_rate_hz)
            self._name = name
            self._subsystem = subsystem

        # TelemetryDataset Protocol surface
        self._channels = (
            Channel(
                name="command_token",
                units=Units.COUNT,
                subsystem=self._subsystem,
                sample_rate_hz=self._sample_rate_hz,
                description="Command-stream token id (multi-token encoding)",
            ),
        )
        # tensor view: (N, 1) float32 — tokens cast for Protocol compliance
        self._data = self._tokens.astype(np.float32).reshape(-1, 1)
        self._timestamps = np.arange(len(self._tokens), dtype=np.int64) * int(
            1000 / self._sample_rate_hz
        )

    # ── TelemetryDataset Protocol ────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def subsystem(self) -> Subsystem:
        return self._subsystem

    @property
    def channels(self) -> tuple[Channel, ...]:
        return self._channels

    @property
    def sample_rate_hz(self) -> float:
        return self._sample_rate_hz

    @property
    def has_labels(self) -> bool:
        return True

    @property
    def tokenizer(self) -> CommandTokenizer | None:
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is not None:
            return self._tokenizer.vocab_size
        return int(self._tokens.max()) + 1

    def windows(self, length: int, stride: int) -> Iterator[TelemetryWindow]:
        n = len(self._tokens)
        if length > n:
            return
        for start in range(0, n - length + 1, stride):
            end = start + length
            aux = None
            if self._aux is not None:
                aux = {k: v[start:end].copy() for k, v in self._aux.items()}
            yield TelemetryWindow(
                tensor=self._data[start:end].copy(),
                timestamps=self._timestamps[start:end].copy(),
                channels=self._channels,
                labels=self._labels[start:end].copy(),
                aux_labels=aux,
                provenance={"source": self._name, "start": start, "end": end},
            )

    def split(self, scheme: SplitScheme) -> Mapping[str, CommandSequenceDataset]:
        n = len(self._tokens)
        n_train = int(scheme.train_frac * n)
        n_val = int(scheme.val_frac * n)
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n),
        }
        out: dict[str, CommandSequenceDataset] = {}
        for split_name, (s, e) in bounds.items():
            sub_aux = (
                {k: v[s:e].copy() for k, v in self._aux.items()}
                if self._aux is not None else None
            )
            out[split_name] = CommandSequenceDataset(
                tokens=self._tokens[s:e].copy(),
                labels=self._labels[s:e].copy(),
                aux=sub_aux,
                tokenizer=self._tokenizer,
                sample_rate_hz=self._sample_rate_hz,
                name=f"{self._name}_{split_name}",
                subsystem=self._subsystem,
            )
        return out

    def stats(self) -> DatasetStats:
        return DatasetStats(
            means=self._data.mean(axis=0).astype(np.float32),
            stds=(self._data.std(axis=0) + 1e-8).astype(np.float32),
            p99=np.quantile(np.abs(self._data), 0.99, axis=0).astype(np.float32),
            n_samples=int(self._data.shape[0]),
            n_channels=1,
        )


@DATASET_REGISTRY.register("command_sequence")
def _create(**kwargs: Any) -> CommandSequenceDataset:
    return CommandSequenceDataset(**kwargs)
