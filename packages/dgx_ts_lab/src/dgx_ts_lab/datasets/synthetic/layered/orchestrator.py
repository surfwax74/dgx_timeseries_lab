"""LayeredSyntheticDataset — composes Components into a TelemetryDataset.

Materializes the entire dataset eagerly on construction (in-memory), then
serves windows / splits / stats through the standard TelemetryDataset
Protocol. Configuration arrives as a list of Components (either fully
constructed Python objects or Hydra dicts with ``_target_``).
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

from .component import Component, GenState


class LayeredSyntheticDataset:
    """A TelemetryDataset assembled from a list of Components.

    Parameters
    ----------
    channels : tuple[Channel, ...]
        Channel definitions; order determines the column order in tensors.
    components : list[Component]
        L1–L6 components applied in order. Order matters: physics first,
        modes early so downstream components can branch on them, faults
        last so labels reflect the full additive picture.
    n_samples : int
    sample_rate_hz : float
    seed : int
    name : str
    subsystem : Subsystem
    """

    def __init__(
        self,
        channels: tuple[Channel, ...],
        components: list[Component],
        n_samples: int,
        sample_rate_hz: float = 1.0,
        seed: int = 0,
        name: str = "layered_synth",
        subsystem: Subsystem = Subsystem.UNKNOWN,
        # Phase 6: when True, windows() will populate aux_labels with
        # multi-task targets (fault_type, rul, next_mode) derived from
        # fault_log + mode_trace.
        emit_multitask_labels: bool = False,
        next_mode_horizon_s: float = 60.0,
        _materialized: dict | None = None,
    ) -> None:
        self._channels = tuple(channels)
        self._sample_rate_hz = float(sample_rate_hz)
        self._name = name
        self._subsystem = subsystem
        self._seed = int(seed)
        self._emit_multitask = bool(emit_multitask_labels)
        self._next_mode_horizon_s = float(next_mode_horizon_s)
        self._label_computer = None

        if _materialized is not None:
            # Sub-dataset path from split() — reuse pre-computed arrays.
            self._data = _materialized["data"]
            self._timestamps = _materialized["timestamps"]
            self._labels = _materialized["labels"]
            self._mode = _materialized["mode"]
            self._fault_log = _materialized.get("fault_log", [])
            return

        rng = np.random.default_rng(self._seed)
        t = np.arange(n_samples, dtype=np.float32) / self._sample_rate_hz
        state = GenState(
            t=t,
            data=np.zeros((n_samples, len(self._channels)), dtype=np.float32),
            mode=np.full(n_samples, -1, dtype=np.int32),
            labels=np.zeros(n_samples, dtype=np.bool_),
            channel_index={c.name: i for i, c in enumerate(self._channels)},
            sample_rate_hz=self._sample_rate_hz,
        )
        for component in components:
            component.apply(state, rng)

        # Stash the components so the Phase 7 explanation layer can extract
        # a ground-truth coupling graph for cascade analysis.
        self._components = list(components)
        self._data = state.data
        self._mode = state.mode
        self._labels = state.labels
        self._fault_log = state.fault_log
        self._timestamps = (t * 1000.0).astype(np.int64)  # epoch ms (relative)

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
    def fault_log(self) -> list[dict[str, Any]]:
        return list(self._fault_log)

    @property
    def mode_trace(self) -> np.ndarray:
        return self._mode

    def _ensure_label_computer(self) -> None:
        if not self._emit_multitask:
            return
        if self._label_computer is not None:
            return
        from .labels import MultiTaskLabelComputer

        self._label_computer = MultiTaskLabelComputer(
            fault_log=self._fault_log,
            mode_trace=self._mode,
            sample_rate_hz=self._sample_rate_hz,
            next_mode_horizon_s=self._next_mode_horizon_s,
        )

    def windows(self, length: int, stride: int) -> Iterator[TelemetryWindow]:
        n = self._data.shape[0]
        if length > n:
            return
        self._ensure_label_computer()
        for start in range(0, n - length + 1, stride):
            end = start + length
            aux = None
            if self._emit_multitask and self._label_computer is not None:
                aux = self._label_computer.labels_for_window(start, length)
            yield TelemetryWindow(
                tensor=self._data[start:end].copy(),
                timestamps=self._timestamps[start:end].copy(),
                channels=self._channels,
                labels=self._labels[start:end].copy(),
                aux_labels=aux,
                provenance={"source": self._name, "start": start, "end": end},
            )

    def split(self, scheme: SplitScheme) -> Mapping[str, "LayeredSyntheticDataset"]:
        n = self._data.shape[0]
        n_train = int(scheme.train_frac * n)
        n_val = int(scheme.val_frac * n)
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n),
        }
        out: dict[str, LayeredSyntheticDataset] = {}
        for split_name, (s, e) in bounds.items():
            # Filter the fault log to only those entirely within the slice.
            sub_log = [
                {**f, "start": f["start"] - s, "end": f["end"] - s}
                for f in self._fault_log
                if f.get("start", 0) >= s and f.get("end", 0) <= e
            ]
            out[split_name] = LayeredSyntheticDataset(
                channels=self._channels,
                components=[],  # ignored — we pass _materialized
                n_samples=e - s,
                sample_rate_hz=self._sample_rate_hz,
                seed=self._seed,
                name=f"{self._name}_{split_name}",
                subsystem=self._subsystem,
                _materialized={
                    "data": self._data[s:e].copy(),
                    "timestamps": self._timestamps[s:e].copy(),
                    "labels": self._labels[s:e].copy(),
                    "mode": self._mode[s:e].copy(),
                    "fault_log": sub_log,
                },
            )
        return out

    def stats(self) -> DatasetStats:
        return DatasetStats(
            means=self._data.mean(axis=0).astype(np.float32),
            stds=(self._data.std(axis=0) + 1e-8).astype(np.float32),
            p99=np.quantile(np.abs(self._data), 0.99, axis=0).astype(np.float32),
            n_samples=int(self._data.shape[0]),
            n_channels=int(self._data.shape[1]),
        )


def _materialize_channel(c: Any) -> Channel:
    if isinstance(c, Channel):
        return c
    if isinstance(c, Mapping):
        d = dict(c)
        if "units" in d and not isinstance(d["units"], Units):
            d["units"] = Units(d["units"])
        if "subsystem" in d and not isinstance(d["subsystem"], Subsystem):
            d["subsystem"] = Subsystem(d["subsystem"])
        return Channel(**d)
    raise TypeError(f"cannot coerce {type(c).__name__} to Channel")


def _materialize_component(c: Any) -> Component:
    if isinstance(c, Component):
        return c
    if isinstance(c, Mapping):
        # Lazy import to keep core import path lightweight when Hydra isn't used.
        from hydra.utils import instantiate

        return instantiate(c)
    raise TypeError(f"cannot coerce {type(c).__name__} to Component")


@DATASET_REGISTRY.register("layered_synth")
def _create(
    *,
    channels: list,
    components: list,
    n_samples: int,
    sample_rate_hz: float = 1.0,
    seed: int = 0,
    name: str = "layered_synth",
    subsystem: str | Subsystem = Subsystem.UNKNOWN,
) -> LayeredSyntheticDataset:
    sub = subsystem if isinstance(subsystem, Subsystem) else Subsystem(subsystem)
    ch_objs = tuple(_materialize_channel(c) for c in channels)
    comp_objs = [_materialize_component(c) for c in components]
    return LayeredSyntheticDataset(
        channels=ch_objs,
        components=comp_objs,
        n_samples=n_samples,
        sample_rate_hz=sample_rate_hz,
        seed=seed,
        name=name,
        subsystem=sub,
    )
