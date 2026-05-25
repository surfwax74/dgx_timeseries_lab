# configs/trainer/

Hydra group for training configuration. Maps directly onto `dgx_ts_core.training.TrainConfig`.

## Files

## Tiered configs (Phase 4-prep)

| YAML | Hardware | Approx VRAM | Realistic model ceiling |
|---|---|---|---|
| `cpu.yaml` | CPU only | RAM-bound | ~5M params, window ≤256 |
| `rtx3080.yaml` | 1× RTX 3080 (12 GB) | 12 GB | ~50M fp32, ~150M bf16 |
| `rtx3080_x2.yaml` | 2× RTX 3080 (DDP, GLOO) | 24 GB pooled | ~100M bf16 |
| `a5000.yaml` | 1× A5000 (24 GB) | 24 GB | ~200M bf16 |
| `a5000_x8.yaml` | 8× A5000 (DDP/FSDP) | 192 GB pooled | ~500M FSDP |
| `h200.yaml` | 1× H200 (141 GB) | 141 GB | ~7B bf16 |
| `h200_fsdp_8x.yaml` | 8× H200 FSDP | ~1.1 TB | 70B+ with activation checkpointing |

Run a tier-aware launch script (see [`scripts/README.md`](../../scripts/README.md)) to invoke the right config automatically, or pass explicitly:

```powershell
dgx-ts train experiment=phase2_bakeoff trainer=rtx3080
```

**Legacy aliases**: `single_cpu.yaml` and `single_h200.yaml` are kept for back-compat with phase0/1 experiments. New work should use the tiered names.

## What each tier does differently

| Knob | CPU | RTX 3080 | A5000 | H200 |
|---|---|---|---|---|
| `batch_size` | 16 | 32 | 64 | 256 |
| `window_length` | 128 | 256 | 512 | 2048–4096 |
| `precision` | 32-true | bf16-mixed | bf16-mixed | bf16-mixed |
| `strategy` | auto | auto | auto / fsdp | fsdp |
| `num_workers` | 0 | 4 | 8 | 16 |

## See also

- Hardware compatibility matrix: [`docs/deployment/hardware_compatibility_matrix.md`](../../docs/deployment/hardware_compatibility_matrix.md)
- Per-tier playbooks: [`docs/deployment/`](../../docs/deployment/README.md)

## Original Phase 0–3 configs

| YAML | Device / strategy | Typical use |
|---|---|---|
| `single_cpu.yaml` | CPU | Legacy alias; same as `cpu.yaml`. Used by phase0/1 experiments. |
| `single_h200.yaml` | One H200 GPU, bf16-mixed | Legacy alias; same as `h200.yaml`. |
| `fsdp_8xh200.yaml` *(Phase 4)* | 8×H200 FSDP, bf16-mixed | Legacy alias; same as `h200_fsdp_8x.yaml`. |
| `deepspeed_zero3.yaml` *(Phase 4)* | DeepSpeed ZeRO-3 | Alternative to FSDP for very large models. |

## Key fields

- `max_epochs`, `batch_size`, `window_length`, `window_stride` — training shape.
- `learning_rate`, `seed` — standard knobs.
- `device`: `auto | cpu | cuda`.
- `precision`: passed to Lightning Fabric (`32-true`, `bf16-mixed`, `16-mixed`, …).
- `strategy`: `auto | ddp | fsdp | deepspeed_stage_*`.
- `checkpoint_dir`: where the detector's `save()` writes.

## Overriding from CLI

```powershell
dgx-ts train experiment=phase0_smoke trainer.max_epochs=5 trainer.batch_size=256
```
