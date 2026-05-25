# dgx_ts_core

Pure interfaces for time-series anomaly detection. No torch, no lightning,
no MLflow — only `numpy` and `typing-extensions`.

This package is the **lift-to-MLOps boundary**. Downstream systems
(`mm_mlops`, inference servers, monitoring tools) pip-install this package
to consume exported model artifacts without dragging training-time
dependencies into their runtime.

## What lives here

```
dgx_ts_core/
├── data/
│   ├── window.py       # TelemetryWindow — the only type crossing dataset↔model
│   ├── schema.py       # Channel, Subsystem, Units
│   ├── dataset.py      # TelemetryDataset Protocol, DatasetStats
│   └── splits.py       # SplitScheme, SplitStrategy
├── models/
│   ├── detector.py     # AnomalyDetector Protocol, FitResult
│   ├── capabilities.py # Capabilities, FitMode, OutputKind
│   └── scores.py       # AnomalyScore
├── training/
│   ├── trainer.py      # Trainer Protocol
│   └── config.py       # TrainConfig
├── evaluation/
│   ├── metrics.py      # Metric Protocol
│   └── result.py       # EvalReport
├── export/
│   ├── model_card.py   # ModelCard
│   ├── feature_schema.py
│   └── formats.py      # ExportFormat enum
└── registry.py         # Registry primitive + global DATASET/DETECTOR/TRAINER registries
```

## Invariants

- **No deep-learning framework imports.** `tests/test_no_torch_dep.py`
  enforces this at CI time. Adding `torch`, `lightning`, `mlflow`, or
  `hydra` to this package's runtime imports is a regression.
- **All data containers are immutable dataclasses with `slots=True`.** Cheap
  to construct, hashable where appropriate, and safe to share across
  threads/processes.
- **All interfaces are `Protocol` (structural)**, not `ABC` (nominal). New
  implementations don't need to inherit from anything — they just need to
  match the shape.

## Tests

```powershell
uv run pytest packages/dgx_ts_core/
```
