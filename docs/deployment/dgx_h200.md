# DGX 8×H200 deployment

The flex tier. 8× H200 = 1.1 TB pooled HBM3e — enough for 1B+ parameter from-scratch Sat-TSFM training under FSDP, or full fine-tuning of any foundation model in the lab.

## What works

| Component | Status |
|---|---|
| Everything in Phase 0–3 | ✓ |
| Sat-TSFM 500 M | ✓ Single H200 |
| Sat-TSFM 1 B+ | ✓ FSDP |
| Full fine-tune of MOMENT-large / Moirai-large | ✓ |
| Long context (4 K – 32 K timesteps) | ✓ |
| Multi-day training runs | ✓ |

## Prereqs

- DGX OS or RHEL with NVIDIA driver ≥ 545 (H200 requires recent driver)
- CUDA 12.6+ (12.8 recommended)
- NCCL with H200-aware build
- Python 3.12 + uv
- For air-gap: see [`../air_gapped_setup.md`](../air_gapped_setup.md)

## Install

```bash
git clone <repo-url>   # or sneakernet
cd dgx_timeseries_lab
uv sync

# Air-gap: scripts/setup_dgx.sh handles offline sync from cached wheelhouse
./scripts/setup_dgx.sh

# CUDA torch (cu128 for H200):
./scripts/install_cuda_torch.sh 128
export UV_NO_SYNC=1

# Verify
uv run --no-sync python scripts/check_gpu.py
```

Expected:
```
== GPU preflight ==
  [0] NVIDIA H200                   VRAM 141.0 GB total (140.0 GB free) cc 9.0
  [1] NVIDIA H200                   VRAM 141.0 GB total (140.0 GB free) cc 9.0
  ...
  [7] NVIDIA H200                   VRAM 141.0 GB total (140.0 GB free) cc 9.0
  Recommended trainer config: trainer=h200_fsdp_8x
```

## Standard launches

### Phase 2 bake-off (sanity)
```bash
./scripts/launch_dgx_h200.sh phase2_bakeoff 30
```
The 4 detectors against the 83-ch preset, much higher epoch count than CPU/RTX. Should produce >0.95 ROC-AUC numbers across the board with proper training.

### Phase 3 foundation bake-off
```bash
# After sneakernetting weights into data/models/ and (optionally) registering with MLflow:
./scripts/setup_mlflow_registry.sh &     # background server
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
./scripts/register_foundation_models.py

./scripts/launch_dgx_h200.sh phase3_bakeoff 50
```

### Phase 4 scale training (the headline)
```bash
# Sat-TSFM 1B+ params from scratch under FSDP
./scripts/launch_dgx_h200.sh phase4_scale 200
```

Run a multi-day training. Checkpoints to `checkpoints/`; MLflow logs metrics every epoch.

## Trainer config (`configs/trainer/h200_fsdp_8x.yaml`)

```yaml
batch_size: 32              # per-device; effective = 256
window_length: 4096
precision: bf16-mixed
strategy: fsdp
extra:
  grad_clip_norm: 1.0
  fsdp_auto_wrap_min_params: 100000000   # 100M
  fsdp_activation_checkpointing: true
```

For a single H200 use `trainer=h200` instead.

## NCCL tuning for H200 fabric

The launch script sets sane defaults:
```bash
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL        # use NVLink between H200s
export NCCL_IB_DISABLE=0          # use InfiniBand if present
```

For multi-node (not yet shipped — Phase 5+), add `NCCL_SOCKET_IFNAME`.

## Throughput profiling

```bash
uv run --no-sync python scripts/profile_throughput.py model=sat_tsfm trainer=h200_fsdp_8x
```

Reports tokens/sec, peak GPU memory per device, MFU (model FLOPs utilization). Target: ≥40% MFU on Sat-TSFM @ 500 M with bf16-mixed FSDP.

## Air-gapped operation

- Pre-cache wheels: `uv pip download -d /path/to/wheelhouse -r requirements.txt` on a connected machine, sneakernet over.
- Foundation weights: see [`../foundation_model_provisioning.md`](../foundation_model_provisioning.md)
- NASA datasets: see [`../air_gapped_setup.md`](../air_gapped_setup.md)
- MLflow Registry: stays local (sqlite + filesystem backend)

## Common issues

| Symptom | Fix |
|---|---|
| `check_gpu.py` says "No CUDA GPUs" | Driver ≥ 545 needed; check `nvidia-smi` reports H200 |
| NCCL hangs at start | `export NCCL_DEBUG=INFO`; check H200 fabric and `nvidia-smi nvlink -s` |
| FSDP OOM at expected size | Enable activation checkpointing (`extra.fsdp_activation_checkpointing: true`); reduce `batch_size` |
| MFU < 30% | Increase `window_length`; verify activation checkpointing isn't accidentally on for small models |
| MLflow Registry fetches slow | Move artifact store to NVMe; check `du -sh mlflow_registry/artifacts/` |

## See also

- Per-tier playbook overview: [`README.md`](README.md)
- Hardware compatibility matrix: [`hardware_compatibility_matrix.md`](hardware_compatibility_matrix.md)
- Phase 4 plan (when published): `docs/phase_plans/phase4.md`
