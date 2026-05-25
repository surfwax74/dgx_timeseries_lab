# dgx_ts_core.training

Training-side contracts. Concrete trainers live in `dgx_ts_lab.training`.

## Files

| File | What it defines |
|---|---|
| `trainer.py` | `Trainer` Protocol — `.fit(detector, dataset, mode, config)` and `.zero_shot(...)`. |
| `config.py` | `TrainConfig` dataclass — framework-agnostic training config (epochs, batch size, window length, strategy, precision, …). |
| `__init__.py` | Re-exports. |

## Conventions

- `TrainConfig` is intentionally generic. Concrete Trainer implementations may subclass to add framework-specific fields (e.g., FSDP shard size).
- Hydra YAMLs in `configs/trainer/` map onto `TrainConfig` field names.
- A Trainer must handle three `FitMode` values: `PRETRAIN`, `FINETUNE`, `ZEROSHOT`. The Lightning trainer in `dgx_ts_lab` branches on these.

## See also

- Parent: [`../README.md`](../README.md)
- Concrete trainer: [`packages/dgx_ts_lab/src/dgx_ts_lab/training/README.md`](../../../../dgx_ts_lab/src/dgx_ts_lab/training/README.md)
