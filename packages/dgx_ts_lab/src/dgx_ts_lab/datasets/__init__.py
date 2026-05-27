"""Dataset implementations. Importing this module registers all bundled
datasets with dgx_ts_core.registry.DATASET_REGISTRY."""

from . import (  # noqa: F401  side-effect: register all bundled datasets
    cyber,
    multimodal,
    nasa_telemanom,
    parquet_telemetry,
    synthetic,
)
from .nasa_telemanom import NasaTelemanomChannel
from .parquet_telemetry import ParquetTelemetryDataset

__all__ = [
    "NasaTelemanomChannel",
    "ParquetTelemetryDataset",
    "cyber",
    "multimodal",
    "nasa_telemanom",
    "parquet_telemetry",
    "synthetic",
]
