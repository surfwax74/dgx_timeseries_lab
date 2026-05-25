# RTX 3080 workstation deployment

Use this tier for: real training of small-to-medium models on a single consumer GPU. The RTX 3080 (10–12 GB VRAM, compute capability 8.6) handles Phase 2 from-scratch detectors comfortably and Phase 3 LoRA fine-tuning of small foundation models.

Also applies to: RTX 3080 Ti, RTX 3090, RTX 4080, RTX 4090, A4000 (similar VRAM tier).

## What works

| Component | Status | Notes |
|---|---|---|
| Phase 0–3 datasets + bake-off | ✓ | LEO EPS full preset (83 ch) runs fine |
| `rolling_mean` / `anomaly_transformer` / `dcdetector` / `patchtst_mae` | ✓ | Train at sane speeds (seconds-to-minutes per epoch) |
| `chronos` / `moment` zero-shot + LoRA (small variants) | ✓ | Chronos-tiny + LoRA fits easily |
| `chronos-base` / `moment-base` full fine-tune | ⚠ Tight | Likely needs bf16-mixed + small batch |
| Sat-TSFM ≤50M params | ✓ Phase 4 ready | |
| Sat-TSFM ≥200M | ✗ | Won't fit in 12 GB |
| FSDP | ✗ Not useful | One GPU; use single-device strategy |
| DDP 2-GPU | ✓ | If you have 2× RTX 3080, use `trainer=rtx3080_x2` |

## Prereqs

- Windows 10/11 or Linux
- NVIDIA driver ≥ 525 (verify `nvidia-smi` shows your GPU and a CUDA Version)
- RTX 3080 with at least 8 GB free (close games/browsers when training)
- Python 3.12 + uv installed

## The CUDA torch dance

`uv sync` pulls the CPU-only torch wheel. You need the CUDA wheel for the GPU to be visible:

```powershell
# Once:
.\scripts\install_cuda_torch.ps1            # cu126 default (NVIDIA driver 525+)
# or for newer drivers (driver 555+):
.\scripts\install_cuda_torch.ps1 -Cuda 128

# Every shell session:
$env:UV_NO_SYNC = '1'
```

`UV_NO_SYNC=1` is critical — without it, every `uv run` reverts torch to the CPU wheel.

## Verify

```powershell
uv run --no-sync python scripts/check_gpu.py
```

Expected output:
```
== GPU preflight ==
  [0] NVIDIA GeForce RTX 3080        VRAM  12.0 GB total ( 10.9 GB free) cc 8.6
  Recommended trainer config: trainer=rtx3080
```

## Smoke commands

```powershell
# All-in-one launch script:
.\scripts\launch_rtx3080.ps1

# Or step-by-step:
uv run --no-sync dgx-ts synth dataset=presets/leo_eps_full_24h
uv run --no-sync dgx-ts benchmark experiment=phase2_bakeoff trainer=rtx3080
```

The bake-off runs the four detectors (rolling_mean + 3 from-scratch transformers) against the 83-channel LEO EPS preset. Expected runtime: ~5–10 minutes for 5 epochs. Report at `benchmark_reports/phase2_bakeoff_leo_eps_full/benchmark_report.md`.

## Trainer config knobs (`configs/trainer/rtx3080.yaml`)

```yaml
batch_size: 32
window_length: 256
precision: bf16-mixed       # RTX 3080 supports bf16 (Ampere)
strategy: auto              # single-device, no FSDP/DDP
num_workers: 4
extra:
  grad_clip_norm: 1.0
```

If a model OOMs, halve `batch_size` first, then `window_length`. If you have 2× RTX 3080, switch to `trainer=rtx3080_x2` (DDP, batch effectively doubles).

## Phase 3 with foundation models

Get the Chronos-tiny weights first (one-time, on a connected machine):

```powershell
huggingface-cli download amazon/chronos-t5-tiny --local-dir data\models\amazon\chronos-t5-tiny
```

Then:

```powershell
uv run --no-sync dgx-ts train `
    model=chronos_zero `
    dataset=parquet `
    trainer=rtx3080 `
    mode=zeroshot

# Or LoRA fine-tune:
uv run --no-sync dgx-ts train `
    model=chronos_lora `
    dataset=parquet `
    trainer=rtx3080 `
    mode=finetune `
    +trainer.extra.use_lora=true
```

## Common issues

| Symptom | Fix |
|---|---|
| `check_gpu.py` says "No CUDA GPUs detected" | Run `install_cuda_torch.ps1` and `$env:UV_NO_SYNC = '1'` |
| OOM during training | Halve `trainer.batch_size`, then `trainer.window_length` |
| `nvidia-smi` doesn't show RTX 3080 | Driver/Windows issue — reinstall NVIDIA driver |
| 2× RTX 3080 DDP hangs on Windows | Use `gloo` backend: `$env:TORCH_DISTRIBUTED_BACKEND = 'gloo'` (NCCL is Linux-only) |
| Throughput much lower than expected | Check `nvidia-smi` for other GPU processes (browsers, games); reduce `num_workers` if disk-bottlenecked |

## Next tier up

When you need bigger models (> 50 M) or multi-GPU FSDP: [`a5000_server.md`](a5000_server.md).
