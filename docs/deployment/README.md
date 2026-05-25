# Deployment playbooks

The lab is designed to scale from a CPU-only laptop up to a full DGX 8×H200. Pick the playbook that matches your hardware and follow the steps. Each playbook covers prereqs, install verification, recommended models, a smoke run, and common issues.

## Tiers

| Tier | Hardware | Playbook | When to use |
|---|---|---|---|
| **CPU-only** | Any modern CPU | [`cpu_only.md`](cpu_only.md) | CI, dev iteration on the protocol layer, "does it run?" sanity |
| **RTX 3080 workstation** | 1–2× RTX 3080 / 3090 / 4080 / 4090 (10–24 GB each) | [`rtx3080_workstation.md`](rtx3080_workstation.md) | Real training of small models (~50 M params); validating the Fabric loop with a GPU |
| **A5000 server** | 1–8× A5000 / A6000 (24–48 GB each) | [`a5000_server.md`](a5000_server.md) | Medium models (~200–500 M), full Phase 2/3 bake-offs at usable speeds |
| **DGX 8×H200** | 8× H200 (141 GB each, ~1.1 TB pooled) | [`dgx_h200.md`](dgx_h200.md) | Phase 4 scale story — 1 B+ param Sat-TSFM, full FSDP, long-context experiments |

## Hardware compatibility matrix

What model × tier combinations actually fit and complete in reasonable time? See [`hardware_compatibility_matrix.md`](hardware_compatibility_matrix.md) — quick reference for picking experiments.

## The launch-script pattern

Every tier ships a launch script under `scripts/`:

| Script | Purpose |
|---|---|
| `launch_cpu_smoke.ps1` / `.sh` | Phase 0 + Phase 1 smoke on CPU |
| `launch_rtx3080.ps1` | Phase 2 bake-off on 1× RTX 3080 |
| `launch_a5000.sh` | Auto-detects 1 vs N A5000s; runs phase 2 or 3 |
| `launch_dgx_h200.sh` | Auto-detects 1 vs 8 H200s; runs phase 4 |

Each script does the same three things:

1. **Preflight** — `scripts/check_gpu.py` reports detected hardware and rejects mismatched configs.
2. **Materialize data** — runs `dgx-ts synth` if the parquet dataset isn't already on disk.
3. **Launch** — runs `dgx-ts benchmark` (or `train`) with the right `trainer=<tier>` config.

## The uv + CUDA Windows gotcha

`uv sync` installs the CPU-only torch wheel by default. CUDA users need to swap it:

```powershell
# Once after `uv sync`:
.\scripts\install_cuda_torch.ps1

# Then set this in every shell so `uv run` doesn't re-sync away your CUDA wheel:
$env:UV_NO_SYNC = '1'
```

Same on Linux: `./scripts/install_cuda_torch.sh` + `export UV_NO_SYNC=1`. Covered in every tier playbook.

## See also

- Trainer configs reference: [`configs/trainer/README.md`](../../configs/trainer/README.md)
- GPU preflight script: [`scripts/check_gpu.py`](../../scripts/check_gpu.py)
- Air-gap setup: [`../air_gapped_setup.md`](../air_gapped_setup.md)
