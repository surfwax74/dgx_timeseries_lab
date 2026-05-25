"""NASA Telemanom (SMAP / MSL) benchmark loader.

Air-gap aware: never tries to download. Expects local files at ``data_root``.
If files are missing, raises with a clear pointer at where to obtain them.

Telemanom dataset layout (per https://github.com/khundman/telemanom):

    data_root/
      labeled_anomalies.csv         # chan_id, spacecraft, anomaly_sequences, …
      train/<chan_id>.npy           # (N, F) — col 0 is the telemetry value
      test/<chan_id>.npy            # (M, F) — anomalies present in col 0

Each channel becomes one ``NasaTelemanomChannel`` dataset. ``split()`` returns
the canonical Telemanom partition (train.npy → train, test.npy → val+test);
the input ``SplitScheme`` fractions are honored within the test data only.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
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

_EXPECTED_LAYOUT = """\
Expected layout at data_root:
    labeled_anomalies.csv
    train/<chan_id>.npy
    test/<chan_id>.npy

Obtain (on a connected machine, then sneakernet to the air-gapped DGX):
    git clone https://github.com/khundman/telemanom
or:
    curl -O https://s3-us-west-2.amazonaws.com/telemanom/data.zip
"""


def _read_anomaly_intervals(
    csv_path: Path, chan_id: str, spacecraft: str
) -> list[tuple[int, int]]:
    """Find this channel's anomaly intervals in labeled_anomalies.csv."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["chan_id"] == chan_id and row["spacecraft"].upper() == spacecraft.upper():
                # e.g. "[[2149, 2349], [4536, 4844]]"
                intervals = json.loads(row["anomaly_sequences"])
                return [(int(s), int(e)) for s, e in intervals]
    raise KeyError(
        f"channel '{chan_id}' (spacecraft={spacecraft}) not found in {csv_path}"
    )


class NasaTelemanomChannel:
    """One channel from NASA SMAP or MSL benchmark, loaded from local files."""

    def __init__(
        self,
        data_root: str | Path,
        channel_id: str,
        spacecraft: str = "SMAP",
        include_commands: bool = False,
        sample_rate_hz: float = 1.0,
        _materialized: dict[str, Any] | None = None,
    ) -> None:
        self._channel_id = channel_id
        self._spacecraft = spacecraft.upper()
        self._sample_rate_hz = float(sample_rate_hz)
        self._include_commands = bool(include_commands)

        if _materialized is not None:
            self._train_data = _materialized["train_data"]
            self._test_data = _materialized["test_data"]
            self._train_labels = _materialized["train_labels"]
            self._test_labels = _materialized["test_labels"]
            self._channels = _materialized["channels"]
            self._name = _materialized["name"]
            self._subsystem = _materialized.get("subsystem", Subsystem.UNKNOWN)
            return

        root = Path(data_root)
        csv_path = root / "labeled_anomalies.csv"
        train_path = root / "train" / f"{channel_id}.npy"
        test_path = root / "test" / f"{channel_id}.npy"

        for p in (csv_path, train_path, test_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"NASA Telemanom file missing: {p}\n\n{_EXPECTED_LAYOUT}"
                )

        intervals = _read_anomaly_intervals(csv_path, channel_id, self._spacecraft)
        train_arr = np.load(train_path)
        test_arr = np.load(test_path)
        if train_arr.ndim == 1:
            train_arr = train_arr.reshape(-1, 1)
        if test_arr.ndim == 1:
            test_arr = test_arr.reshape(-1, 1)

        if not include_commands:
            train_arr = train_arr[:, :1]
            test_arr = test_arr[:, :1]

        self._train_data = train_arr.astype(np.float32)
        self._test_data = test_arr.astype(np.float32)
        self._train_labels = np.zeros(len(self._train_data), dtype=np.bool_)
        self._test_labels = np.zeros(len(self._test_data), dtype=np.bool_)
        for start, end in intervals:
            end = min(end + 1, len(self._test_labels))
            self._test_labels[start:end] = True

        ch_objs: list[Channel] = [
            Channel(
                name=channel_id,
                units=Units.DIMENSIONLESS,
                subsystem=Subsystem.UNKNOWN,
                sample_rate_hz=self._sample_rate_hz,
                description=f"{self._spacecraft} {channel_id} (telemetry value)",
            )
        ]
        if include_commands:
            for i in range(1, self._train_data.shape[1]):
                ch_objs.append(
                    Channel(
                        name=f"{channel_id}_cmd{i}",
                        units=Units.DIMENSIONLESS,
                        subsystem=Subsystem.UNKNOWN,
                        sample_rate_hz=self._sample_rate_hz,
                        description=f"{self._spacecraft} {channel_id} command {i}",
                    )
                )
        self._channels = tuple(ch_objs)
        self._name = f"nasa_{self._spacecraft.lower()}_{channel_id}"
        self._subsystem = Subsystem.UNKNOWN

    # ── concatenated train+test view for top-level Protocol methods ──

    @property
    def _all_data(self) -> np.ndarray:
        return np.concatenate([self._train_data, self._test_data], axis=0)

    @property
    def _all_labels(self) -> np.ndarray:
        return np.concatenate([self._train_labels, self._test_labels], axis=0)

    @property
    def _all_timestamps(self) -> np.ndarray:
        n = len(self._all_data)
        return (np.arange(n, dtype=np.int64) * int(1000 / self._sample_rate_hz))

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

    def windows(self, length: int, stride: int) -> Iterator[TelemetryWindow]:
        data = self._all_data
        labels = self._all_labels
        timestamps = self._all_timestamps
        n = len(data)
        if length > n:
            return
        for start in range(0, n - length + 1, stride):
            end = start + length
            yield TelemetryWindow(
                tensor=data[start:end].copy(),
                timestamps=timestamps[start:end].copy(),
                channels=self._channels,
                labels=labels[start:end].copy(),
                provenance={"source": self._name, "start": start, "end": end},
            )

    def split(self, scheme: SplitScheme) -> Mapping[str, "NasaTelemanomChannel"]:
        # Canonical Telemanom partition: train.npy → train; test.npy split
        # into val + test by the scheme's val_frac : test_frac ratio.
        n_test = len(self._test_data)
        val_size = int(n_test * scheme.val_frac / (scheme.val_frac + scheme.test_frac))
        splits: dict[str, NasaTelemanomChannel] = {}

        def _mk(name: str, data: np.ndarray, labels: np.ndarray) -> NasaTelemanomChannel:
            return NasaTelemanomChannel(
                data_root="",  # ignored in materialized path
                channel_id=self._channel_id,
                spacecraft=self._spacecraft,
                include_commands=self._include_commands,
                sample_rate_hz=self._sample_rate_hz,
                _materialized={
                    "train_data": data,
                    "test_data": data[:0],  # empty (already-sliced view)
                    "train_labels": labels,
                    "test_labels": labels[:0],
                    "channels": self._channels,
                    "name": f"{self._name}_{name}",
                    "subsystem": self._subsystem,
                },
            )

        splits["train"] = _mk("train", self._train_data.copy(), self._train_labels.copy())
        splits["val"] = _mk(
            "val", self._test_data[:val_size].copy(), self._test_labels[:val_size].copy()
        )
        splits["test"] = _mk(
            "test", self._test_data[val_size:].copy(), self._test_labels[val_size:].copy()
        )
        return splits

    def stats(self) -> DatasetStats:
        data = self._all_data
        return DatasetStats(
            means=data.mean(axis=0).astype(np.float32),
            stds=(data.std(axis=0) + 1e-8).astype(np.float32),
            p99=np.quantile(np.abs(data), 0.99, axis=0).astype(np.float32),
            n_samples=int(data.shape[0]),
            n_channels=int(data.shape[1]),
        )


@DATASET_REGISTRY.register("nasa_smap_channel")
def _create_smap(**kwargs: Any) -> NasaTelemanomChannel:
    kwargs.setdefault("spacecraft", "SMAP")
    return NasaTelemanomChannel(**kwargs)


@DATASET_REGISTRY.register("nasa_msl_channel")
def _create_msl(**kwargs: Any) -> NasaTelemanomChannel:
    kwargs.setdefault("spacecraft", "MSL")
    return NasaTelemanomChannel(**kwargs)
