"""Write a FeatureSchema dataclass to YAML.

The schema mirrors ``TelemetryDataset.channels`` + ``DatasetStats`` so the
inference-side feature pipeline stays in lockstep with training assumptions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from dgx_ts_core.data import Channel, DatasetStats
from dgx_ts_core.export import FeatureSchema


def write_feature_schema(
    *,
    channels: tuple[Channel, ...],
    sample_rate_hz: float,
    window_length: int,
    stats: DatasetStats,
    output_path: Path,
) -> FeatureSchema:
    """Build a FeatureSchema and write it to ``output_path`` as YAML."""
    schema = FeatureSchema(
        channels=channels,
        sample_rate_hz=float(sample_rate_hz),
        window_length=int(window_length),
        normalization_means=stats.means.astype(np.float32),
        normalization_stds=stats.stds.astype(np.float32),
    )

    serialized = {
        "sample_rate_hz": schema.sample_rate_hz,
        "window_length": schema.window_length,
        "channels": [
            {
                "name": ch.name,
                "units": ch.units.value,
                "subsystem": ch.subsystem.value,
                "sample_rate_hz": float(ch.sample_rate_hz),
                "description": ch.description,
            }
            for ch in channels
        ],
        "normalization": {
            "means": schema.normalization_means.tolist(),
            "stds": schema.normalization_stds.tolist(),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(serialized, sort_keys=False))
    return schema
