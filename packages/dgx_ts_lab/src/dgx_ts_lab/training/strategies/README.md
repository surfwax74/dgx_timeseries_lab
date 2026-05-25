# dgx_ts_lab.training.strategies

Distributed training strategy helpers. Lightning Fabric does the heavy lifting; these modules assemble the strategy-specific kwargs (FSDP wrap policy, activation checkpointing, sharding strategy, …) that Fabric needs.

## Files

| File | Strategy | Reads from |
|---|---|---|
| `fsdp.py` | FSDP (full-shard, hybrid-shard, etc.) | `TrainConfig.extra.fsdp_*` |

## How the trainer picks one

`fabric_loop.py` reads `config.strategy`. If it's `"fsdp"`, it calls `build_fsdp_strategy_kwargs(config.extra)` to assemble the auxiliary config. For `"ddp"`, `"deepspeed_stage_*"`, `"auto"`, etc., Fabric handles the defaults itself.

## Convention for new strategies

1. Add a `<name>.py` with a `build_<name>_strategy_kwargs(extra) -> dict` function.
2. Wire a branch in `fabric_loop.py` that detects the strategy string and calls your function.
3. Document the `TrainConfig.extra` knobs at the top of your file.

## See also

- Tier trainer configs: [`configs/trainer/README.md`](../../../../../../configs/trainer/README.md)
- DGX deployment playbook: [`docs/deployment/dgx_h200.md`](../../../../../../docs/deployment/dgx_h200.md)
