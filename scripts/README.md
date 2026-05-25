# scripts/

Setup convenience scripts. Documentation-as-code — these are also a reference for what's needed manually if the script doesn't fit your environment.

## Files

## Setup scripts

| File | Run on | What it does |
|---|---|---|
| `setup_dgx.sh` | DGX (Linux) | One-time setup: verify uv, sync env, create dirs, print next steps. |
| `setup_dev_windows.ps1` | Windows dev box | One-time setup: install uv via winget if missing, sync env, create dirs. |
| `setup_mlflow_registry.sh` | DGX or server | Stand up local MLflow Registry (sqlite backend, foreground server on :5000). |
| `install_cuda_torch.ps1` / `.sh` | Any GPU box | Swap the CPU-only torch wheel for the CUDA wheel. Run after `uv sync`. |
| `download_datasets.py` | **Connected machine only** | Fetches NASA SMAP/MSL datasets. In air-gap, this is reference; transfer the output by sneakernet. |
| `register_foundation_models.py` | DGX or server | One-time: walk `data/models/<org>/<model>/` and register each with MLflow Registry. |

## Preflight & launch scripts

| File | Tier | What it does |
|---|---|---|
| `check_gpu.py` | All | Reports CUDA availability, GPU count, VRAM, recommends a trainer config tier. Used by all launch scripts. |
| `launch_cpu_smoke.ps1` / `.sh` | CPU | Phase 0 + Phase 1 smoke on CPU. ~1 min. |
| `launch_rtx3080.ps1` | 1× RTX 3080 | Phase 2 bake-off vs 83-ch preset. ~5–10 min for 5 epochs. |
| `launch_a5000.sh` | 1–8× A5000 | Auto-detects single vs multi; runs phase 2 or 3. |
| `launch_dgx_h200.sh` | 1–8× H200 | Auto-detects single vs FSDP; runs phase 4 or whatever you pass as `$1`. |
| `profile_throughput.py` *(Phase 4)* | GPU only | Reports tokens/sec, peak GPU memory, MFU. |

## Air-gap notes

- `download_datasets.py` MUST NOT be run on the air-gapped DGX. It exists as documentation of what `data/` should contain when populated.
- `setup_dgx.sh` is safe in air-gap — it doesn't reach out to the network.
