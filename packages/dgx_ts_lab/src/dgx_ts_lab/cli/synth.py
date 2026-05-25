"""``dgx-ts synth`` — materialize a dataset to parquet for air-gapped reuse.

Writes five files into ``output_dir/<dataset_name>/``:

    data.parquet      timestamp_ms + one column per channel
    labels.parquet    timestamp_ms + is_anomaly
    fault_log.json    structured fault entries (empty list for non-synthetic)
    channels.yaml     full Channel metadata
    manifest.yaml     dataset name, subsystem, n_samples, n_channels, version, source config

Then ``parquet_telemetry`` can load the same dataset on any machine without
re-running generation — the canonical air-gapped distribution pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import dgx_ts_lab  # noqa: F401  side-effect: register implementations
from dgx_ts_core.registry import DATASET_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[5]
_CONFIG_DIR = _REPO_ROOT / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def run(cfg: DictConfig) -> None:
    # Hydra changes cwd to outputs/<date>/<time>/. Resolve relative output_dir
    # against the original launch cwd so files land where the user expects.
    output_root_raw = Path(cfg.get("output_dir", "data/synth"))
    output_root = (
        output_root_raw
        if output_root_raw.is_absolute()
        else Path(get_original_cwd()) / output_root_raw
    )
    ds_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    assert isinstance(ds_cfg, dict)
    ds_key = ds_cfg.pop("_target_key")
    dataset = DATASET_REGISTRY.create(ds_key, **ds_cfg)

    dataset_name = getattr(dataset, "name", ds_key)
    out_dir = output_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data, timestamps, labels = _materialize_arrays(dataset)
    channels = dataset.channels

    # data.parquet
    data_columns: dict[str, pa.Array] = {
        "timestamp_ms": pa.array(timestamps, type=pa.int64()),
    }
    for i, ch in enumerate(channels):
        data_columns[ch.name] = pa.array(data[:, i], type=pa.float32())
    pq.write_table(pa.table(data_columns), out_dir / "data.parquet")

    # labels.parquet
    pq.write_table(
        pa.table(
            {
                "timestamp_ms": pa.array(timestamps, type=pa.int64()),
                "is_anomaly": pa.array(labels, type=pa.bool_()),
            }
        ),
        out_dir / "labels.parquet",
    )

    # fault_log.json
    fault_log = list(getattr(dataset, "fault_log", []))
    (out_dir / "fault_log.json").write_text(
        json.dumps(fault_log, indent=2, default=str)
    )

    # channels.yaml
    (out_dir / "channels.yaml").write_text(
        yaml.safe_dump(
            {
                "channels": [
                    {
                        "name": ch.name,
                        "units": ch.units.value,
                        "subsystem": ch.subsystem.value,
                        "sample_rate_hz": float(ch.sample_rate_hz),
                        "description": ch.description,
                    }
                    for ch in channels
                ]
            },
            sort_keys=False,
        )
    )

    # manifest.yaml
    manifest = {
        "name": dataset_name,
        "subsystem": dataset.subsystem.value,
        "sample_rate_hz": float(dataset.sample_rate_hz),
        "n_samples": int(data.shape[0]),
        "n_channels": int(data.shape[1]),
        "n_fault_events": len(fault_log),
        "n_anomaly_steps": int(labels.sum()),
        "anomaly_rate": float(labels.mean()),
        "dgx_ts_lab_version": dgx_ts_lab.__version__,
        "source_config": OmegaConf.to_container(cfg.dataset, resolve=True),
    }
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    print("\n-- Synth complete --")
    print(f"output_dir:     {out_dir}")
    print(f"n_samples:      {data.shape[0]}")
    print(f"n_channels:     {data.shape[1]}")
    print(f"anomaly_steps:  {int(labels.sum())} ({100.0 * float(labels.mean()):.2f}%)")
    print(f"fault_events:   {len(fault_log)}")


def _materialize_arrays(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull the full dataset as numpy arrays.

    Fast path: bundled datasets expose ``_data`` / ``_timestamps`` / ``_labels``.
    Generic path: concatenate non-overlapping windows.
    """
    if hasattr(dataset, "_data") and hasattr(dataset, "_labels"):
        return dataset._data, dataset._timestamps, dataset._labels

    # Generic fallback — works for any TelemetryDataset implementation.
    chunks_d: list[np.ndarray] = []
    chunks_t: list[np.ndarray] = []
    chunks_l: list[np.ndarray] = []
    for w in dataset.windows(length=4096, stride=4096):
        chunks_d.append(w.tensor)
        chunks_t.append(w.timestamps)
        if w.labels is not None:
            chunks_l.append(w.labels)
        else:
            chunks_l.append(np.zeros(w.length, dtype=np.bool_))
    if not chunks_d:
        raise RuntimeError(
            f"dataset '{dataset.name}' produced no windows at length=4096"
        )
    return (
        np.concatenate(chunks_d, axis=0),
        np.concatenate(chunks_t, axis=0),
        np.concatenate(chunks_l, axis=0),
    )


if __name__ == "__main__":
    run()
