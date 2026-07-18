"""Dataset implementations. Importing this module registers all bundled
datasets with dgx_ts_core.registry.DATASET_REGISTRY."""

from . import (  # noqa: F401  side-effect: register all bundled datasets
    cyber,
    multimodal,
    nasa_telemanom,
    parquet_corpus,
    parquet_telemetry,
    synthetic,
)
from .nasa_telemanom import NasaTelemanomChannel
from .parquet_corpus import ParquetTelemetryCorpus
from .parquet_telemetry import ParquetTelemetryDataset

__all__ = [
    "NasaTelemanomChannel",
    "ParquetTelemetryCorpus",
    "ParquetTelemetryDataset",
    "cyber",
    "multimodal",
    "nasa_telemanom",
    "parquet_corpus",
    "parquet_telemetry",
    "synthetic",
]
