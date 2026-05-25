# A5000 server deployment

Use this tier for: real training of medium-sized models on 1–8× A5000 (or A6000) GPUs. Each A5000 has 24 GB VRAM, so multi-GPU pooling via DDP or FSDP gets you to 200 M–500 M parameter regimes.

Also applies to: L40 / L40S, RTX 6000 Ada, A40 (similar VRAM tier).

## What works

| Component | Status | Notes |
|---|---|---|
| All Phase 0–3 work | ✓ | Comfortably |
| Sat-TSFM ≤200 M (single A5000) | ✓ | bf16-mixed fits easily |
| Sat-TSFM 200–500 M | ✓ DDP/FSDP | Use 4–8× A5000 |
| Sat-TSFM ≥500 M | ⚠ Tight | Use H200 instead |
| Full fine-tune of MOMENT-base / Moirai-base | ✓ | Single GPU OK |
| Long context (≥2048) | ✓ | Within VRAM budget |

## Prereqs

- Linux (RHEL/Rocky/Ubuntu typical for ML servers)
- NVIDIA driver ≥ 525
- CUDA 12.6 toolkit (or matching torch wheel CUDA version)
- nccl installed (`sudo apt install libnccl2 libnccl-dev` or RHEL equivalent)
- Python 3.12 + uv

## Install

```bash
git clone <repo-url>
cd dgx_timeseries_lab
uv sync

# Swap to CUDA torch:
./scripts/install_cuda_torch.sh 126    # or 128 for newer drivers / H200-class

# Prevent uv from re-syncing:
export UV_NO_SYNC=1

# Verify:
uv run --no-sync python scripts/check_gpu.py
```

Expected:
```
== GPU preflight ==
  [0] NVIDIA RTX A5000               VRAM  24.0 GB total ( 23.5 GB free) cc 8.6
  [1] NVIDIA RTX A5000               VRAM  24.0 GB total ( 23.5 GB free) cc 8.6
  ...
  Recommended trainer config: trainer=a5000_x8
```

## Launch

```bash
./scripts/launch_a5000.sh phase2_bakeoff 20
```

Auto-detects single vs multi-GPU and picks `trainer=a5000` or `trainer=a5000_x8`.

Manual:
```bash
# Single GPU
uv run --no-sync dgx-ts benchmark experiment=phase2_bakeoff trainer=a5000

# Multi-GPU (DDP for <200M, FSDP for 200M-1B)
uv run --no-sync dgx-ts benchmark experiment=phase3_bakeoff trainer=a5000_x8
```

## Trainer config knobs

`configs/trainer/a5000.yaml` (single GPU):
```yaml
batch_size: 64
window_length: 512
precision: bf16-mixed
strategy: auto
num_workers: 8
```

`configs/trainer/a5000_x8.yaml` (8 GPU FSDP):
```yaml
batch_size: 64              # per-device; effective = 512
window_length: 1024
strategy: fsdp              # use "ddp" for models <= 200M
```

## Phase 3 with foundation models

Get bigger weights:
```bash
huggingface-cli download amazon/chronos-t5-small --local-dir data/models/amazon/chronos-t5-small
huggingface-cli download AutonLab/MOMENT-1-base --local-dir data/models/AutonLab/MOMENT-1-base
```

LoRA fine-tune on the cluster:
```bash
uv run --no-sync dgx-ts train \
    model=chronos_lora model.model=amazon/chronos-t5-small \
    dataset=parquet trainer=a5000_x8 mode=finetune \
    +trainer.extra.use_lora=true \
    +trainer.extra.lora_r=16 \
    +trainer.extra.lora_alpha=32
```

## Switch DDP ↔ FSDP

Rule of thumb:
- Model fits comfortably in one GPU (< 50% VRAM) → **DDP** (faster per-step)
- Model nearly fills one GPU (> 70%) → **FSDP** (shards params)

Edit `configs/trainer/a5000_x8.yaml`:
```yaml
strategy: ddp_find_unused_parameters_true   # or fsdp
```

## MLflow Registry on the server

For multi-user setups, stand up the MLflow Registry once:
```bash
./scripts/setup_mlflow_registry.sh         # listens on http://127.0.0.1:5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
./scripts/register_foundation_models.py    # one-time, after sneakernet
```

Now every `dgx-ts` invocation logs to the central Registry.

## Common issues

| Symptom | Fix |
|---|---|
| `check_gpu.py` shows fewer GPUs than expected | `nvidia-smi -L`; check `CUDA_VISIBLE_DEVICES` env var |
| FSDP slower than DDP | Model probably fits in one GPU — switch to `strategy: ddp` |
| `RuntimeError: NCCL error` | `export NCCL_DEBUG=INFO` and retry; check `libnccl2` version |
| OOM at known-good batch | Other processes on the GPU; check `nvidia-smi` |

## Next tier up

For the largest models and the full FSDP scale story: [`dgx_h200.md`](dgx_h200.md).
