# Architecture

## Goal

Run satellite-telemetry anomaly detection experiments on an 8×H200 DGX in an air-gapped environment, with models and datasets that hot-swap behind stable interfaces so working algorithms can be lifted into a larger MLOps platform (`mm_mlops`) without rewrites.

## The two-package shape

```
dgx_timeseries_lab/
├── packages/
│   ├── dgx_ts_core/    ← pure interfaces; no torch
│   └── dgx_ts_lab/     ← implementations; depends on core + torch + lightning + mlflow + hydra
├── configs/            ← Hydra YAMLs
└── experiments/        ← numbered, reproducible run outputs
```

`dgx_ts_core` is the lift-out boundary. Downstream MLOps systems install ONLY this package to consume exported artifacts.

## The three contracts

Hot-swap is built on three small Protocols defined in `dgx_ts_core`:

```
TelemetryWindow            (T, C) float32 + metadata    ← only type crossing dataset↔model
       │
       ▼
TelemetryDataset Protocol  channels, windows(), split(), stats()
       │
       ▼
AnomalyDetector Protocol   fit(), score(), embed(), reconstruct(), save(), load(), export()
       │
       ▼
Trainer Protocol           fit(detector, dataset, mode, config), zero_shot()
```

Training and eval code branches on each detector's declared `Capabilities` — `requires_pretraining`, `supports_streaming`, `output_kind`, `native_context_len`, `supports_peft`, `supports_export_onnx`. Never on `isinstance` of the concrete class.

## Registries

Three global factory registries live in `dgx_ts_core.registry`:

- `DATASET_REGISTRY` (key → factory creating `TelemetryDataset`)
- `DETECTOR_REGISTRY` (key → factory creating `AnomalyDetector`)
- `TRAINER_REGISTRY` (key → factory creating `Trainer`)

Implementations self-register at import time via `@DATASET_REGISTRY.register("my_key")`. Hydra YAMLs reference implementations by key (`_target_key: my_key`), so config sweeps don't require Python edits.

## Run flow

```
configs/experiment/<x>.yaml
   ↓ Hydra composes
{dataset, model, trainer, mode, mlflow.{...}}
   ↓ dgx-ts train
DATASET_REGISTRY.create(ds_key, **ds_cfg)        → TelemetryDataset
DETECTOR_REGISTRY.create(model_key, **model_cfg) → AnomalyDetector
TRAINER_REGISTRY.create(trainer_key)             → Trainer
   ↓
trainer.fit(detector, dataset, mode, config)     → FitResult
   ↓
MLflowLogger logs params, metrics, fit_result.json, detector.<ext>
```

## Phase status

See the [root README](../README.md#phase-plan).

## See also

- Tree walking: every code directory has a `README.md`. Start at the [root](../README.md) and follow the links.
- Lift-out spec: [`lift_to_mlops.md`](lift_to_mlops.md).
- Air-gap provisioning: [`air_gapped_setup.md`](air_gapped_setup.md).
