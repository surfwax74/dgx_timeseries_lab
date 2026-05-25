# dgx_ts_lab

Concrete implementations of the `dgx_ts_core` contracts: datasets,
detectors, trainers, MLflow tracking, and the `dgx-ts` Hydra CLI.

## What lives here

```
dgx_ts_lab/
├── datasets/
│   ├── synthetic/
│   │   ├── trivial.py    # Phase 0: sine + spike (smoke test)
│   │   └── layered/      # Phase 1: L1–L6 composable generator
│   ├── nasa_smap.py      # Phase 1
│   ├── nasa_msl.py       # Phase 1
│   └── ops_sat.py        # Phase 1
├── models/
│   ├── baseline/
│   │   └── rolling_mean.py  # Phase 0: classical z-score detector
│   ├── from_scratch/        # Phase 2: Anomaly Transformer, DCdetector, PatchTST+MAE
│   ├── foundation/          # Phase 3: Chronos, MOMENT, Moirai adapters
│   └── physics/             # Phase 4: PINN residual wrappers
├── training/
│   ├── lightning_trainer.py # Trainer Protocol over Lightning Fabric
│   └── strategies/          # Phase 4: fsdp, deepspeed
├── evaluation/
│   └── metrics.py           # basic_metrics, calibrate_threshold
├── serving/                 # Phase 5: ONNX export, Triton config
├── tracking/
│   └── mlflow_logger.py
└── cli/
    ├── main.py              # Subcommand dispatcher
    └── train.py             # `dgx-ts train` — Hydra-wired training entrypoint
```

## How registration works

Importing `dgx_ts_lab` triggers side-effect imports of `datasets`, `models`,
and `training`. Each bundled implementation registers itself with the
appropriate registry in `dgx_ts_core.registry`. Hydra YAMLs reference
implementations by their registry key (e.g. `_target_key: rolling_mean`),
so config sweeps don't require Python edits.

To add a new dataset / detector / trainer:

1. Implement the relevant Protocol from `dgx_ts_core`.
2. Decorate a factory with `@DATASET_REGISTRY.register("my_key")` (etc.).
3. Add a `configs/<group>/my_key.yaml`.

That's it — `dgx-ts train model=my_key` will pick it up.

## Tests

```powershell
uv run pytest packages/dgx_ts_lab/
```
