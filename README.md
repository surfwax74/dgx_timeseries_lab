# dgx_timeseries_lab

Satellite-telemetry anomaly-detection experiments on an 8×H200 DGX.

The repo is structured so models and datasets can be **hot-swapped** behind
stable abstract interfaces, and so a trained detector can be **lifted into
the broader `mm_mlops` framework** as three artifacts with no Python
back-imports.

## Layout

```
packages/
├── dgx_ts_core/    # Pure interfaces. No torch dependency.
│                   # Downstream MLOps systems consume this package.
└── dgx_ts_lab/     # Implementations: datasets, detectors, trainers,
                    # MLflow tracking, Hydra CLI.
configs/            # Hydra YAMLs (dataset / model / trainer / experiment)
experiments/        # Numbered, reproducible run outputs
data/               # Gitignored; symlink to NAS on DGX
```

## Quickstart

```powershell
uv sync
uv run pytest packages/
uv run dgx-ts train experiment=phase0_smoke
```

The Phase 0 smoke run completes in well under two minutes on CPU, fits
`RollingMeanDetector` against a trivial sine+spike synthetic dataset, and
logs the run to `mlruns/`.

## The three contracts

All hot-swap is built on three small Protocols defined in `dgx_ts_core`:

| Protocol | Purpose |
|---|---|
| `TelemetryWindow` | The only data type crossing dataset↔model boundaries |
| `TelemetryDataset` | Any source: NASA SMAP/MSL, ESA OPS-SAT, synthetic, real mission |
| `AnomalyDetector` | From-scratch transformers, foundation-model adapters, classical baselines, physics residual wrappers — all the same interface |
| `Trainer` | Wraps Lightning Fabric (or any other engine) |

Hot-swap works because training and evaluation branch on a detector's
declared `Capabilities` — `requires_pretraining`, `supports_streaming`,
`output_kind`, `native_context_len`, etc. — never on `isinstance()` of the
concrete class.

## Phase plan

| Phase | Deliverable |
|---|---|
| 0 | Scaffold smoke: protocols, trivial synth, rolling-mean baseline, end-to-end CLI |
| 1 | NASA SMAP/MSL loaders + **layered synthetic generator** (L1–L6 composable components) |
| 2 | From-scratch bake-off: Anomaly Transformer, DCdetector, PatchTST+MAE |
| 3 | Foundation-model adapters: Chronos, MOMENT, Moirai (zero-shot + LoRA) |
| 4 | FSDP 8×H200 training of from-scratch foundation model + PINN residual wrappers |
| 5 | MLOps lift: ONNX export + model card + feature schema |

Each phase ends with a working CLI command, READMEs updated, unit tests
added, and an MLflow experiment showing the result.

## Lift-to-MLOps contract

`detector.export(format=ONNX, path=...)` emits three files that downstream
MLOps systems consume — nothing else from this repo is needed at serving
time:

1. `model.onnx` — runtime artifact
2. `model_card.yaml` — metrics, intended subsystem, calibrated threshold
3. `feature_schema.yaml` — channels, units, sample rate, normalization stats

`feature_schema.yaml` has the same structure as `TelemetryDataset.channels`
+ `DatasetStats`, so training-time and serving-time feature assumptions
stay in lockstep.
