"""dgx_ts_core — pure interfaces for time-series anomaly detection.

This package defines the three contracts that make hot-swap across datasets
and models possible: TelemetryDataset, AnomalyDetector, and Trainer.

It has no torch dependency by design — downstream MLOps systems can import
this package to consume exported model artifacts without dragging training-only
dependencies into their runtime.
"""

__version__ = "0.1.0"
