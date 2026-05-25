# dgx_ts_lab.training

Concrete `Trainer` implementations. Importing this package self-registers every trainer with `dgx_ts_core.registry.TRAINER_REGISTRY`.

## Files

| File | Trainer | Registry key |
|---|---|---|
| `lightning_trainer.py` | `LightningTrainer` — wraps Lightning Fabric. | `lightning` |

## How `LightningTrainer.fit()` branches

It reads `detector.capabilities` to decide the path:

- **`requires_pretraining == False`** (or `mode == ZEROSHOT`): delegates to `detector.fit()` directly. Used by classical baselines like `rolling_mean`.
- **`requires_pretraining == True`**: runs the Lightning Fabric loop (Phase 2 — currently a clearly-marked stub).

After fitting, it always:
1. Calibrates a threshold on the train scores (percentile-based — see `evaluation/metrics.py`).
2. Scores val + test splits.
3. Returns a `FitResult` with `val_metrics` and `test_metrics` in `metadata`.

## Subdirs (planned)

| Subdir | Phase |
|---|---|
| `strategies/` — `single_gpu.py`, `fsdp.py`, `deepspeed.py` | 4 |
| `callbacks/` — checkpoint, LR finder, gradient clipping | 2 |
