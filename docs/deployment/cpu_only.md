# CPU-only deployment

Use this tier for: CI, protocol-layer dev, "does the scaffold run?" sanity. **No real model training happens here** — neural transformers on CPU take minutes per epoch even at tiny sizes.

## What works

| Component | Status | Notes |
|---|---|---|
| All Phase 0 / 1 dataset generators | ✓ Full speed | Synth runs in seconds. |
| `rolling_mean` baseline | ✓ Full speed | Pure numpy; CPU is fine. |
| `anomaly_transformer` / `dcdetector` / `patchtst_mae` | ✓ Works, slow | OK for tiny configs (1–2 layers, d_model=32). |
| `chronos` / `moment` / `moirai` | ✓ Works, very slow | Fine for unit tests; impractical for real training. |
| LoRA fine-tuning | ✓ Works, slow | |
| FSDP / DDP | ✗ N/A | No GPUs to distribute over. |

## Prereqs

- Python 3.12 (`uv python install 3.12` if needed)
- ~5 GB disk for the venv

## Install

```powershell
git clone <repo-url>
cd dgx_timeseries_lab
uv sync
uv run pytest packages/ -q   # 94 tests should pass
```

## Smoke commands

```powershell
.\scripts\launch_cpu_smoke.ps1
```

Or invoke directly:

```powershell
# Phase 0 — trivial synth, rolling_mean baseline
uv run dgx-ts train experiment=phase0_smoke trainer=cpu

# Phase 1 — 6-ch LEO EPS preset
uv run dgx-ts train experiment=phase1_layered trainer=cpu

# Phase 2 smoke — PatchTST+MAE for 2 epochs
uv run dgx-ts train experiment=phase2_smoke trainer=cpu
```

Each completes in well under a minute. Run summaries print to stdout + every run logs to `mlruns/`.

## What the recommended configs do at this tier

The `cpu.yaml` trainer pins:
- `batch_size: 16`, `window_length: 128`, `max_epochs: 2`
- `precision: 32-true` (bf16 mixed precision needs CUDA)
- `num_workers: 0` (avoids fork issues on Windows)

## Common issues

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: torch` | `uv sync` |
| Tests fail with `tests/` import collision | The pytest `--import-mode=importlib` config in `pyproject.toml` should handle this; verify it's set. |
| Hydra fails with `Could not override 'output_dir'` | Use the latest `configs/config.yaml` (has `output_dir` as a struct member). |

## Next tier up

Once CPU smoke passes, validate against your GPU box:

- **One consumer GPU** (RTX 3080/3090/4080/4090) → [`rtx3080_workstation.md`](rtx3080_workstation.md)
- **A5000 server** → [`a5000_server.md`](a5000_server.md)
