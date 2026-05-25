# Air-gapped setup

This repo is designed to run in environments with no internet access. This doc covers what you need to provision ahead of time and how to do it.

## What needs internet (do these on a connected machine)

| Asset | Where to get it | How to transfer |
|---|---|---|
| Python 3.12 + uv | `uv python install 3.12` (or system package manager) | Pre-install on the DGX before disconnect |
| Python wheels (torch, lightning, mlflow, hydra, …) | `uv sync` once on a connected machine, then archive `.venv/` or use `uv pip download` | sneakernet, internal mirror, or a pre-baked container image |
| NASA SMAP / MSL datasets | `git clone https://github.com/khundman/telemanom` or `curl -O https://s3-us-west-2.amazonaws.com/telemanom/data.zip` | sneakernet to `data/nasa_smap/` and `data/nasa_msl/` |
| ESA OPS-SAT data *(Phase 1 future)* | ESA mission data archive | sneakernet to `data/ops_sat/` |
| Foundation-model weights *(Phase 3)* | HuggingFace mirror of Chronos / MOMENT / Moirai / TimesFM | sneakernet to `data/models/<name>/` |

## NASA Telemanom data layout

After transfer, the layout MUST be:

```
data/
└── nasa_smap/
    ├── labeled_anomalies.csv
    ├── train/
    │   ├── A-1.npy
    │   ├── A-2.npy
    │   └── ...
    └── test/
        ├── A-1.npy
        ├── A-2.npy
        └── ...
```

Then configure: `configs/dataset/smap.yaml` already points at `data_root: data/nasa_smap`. Same for `msl.yaml`.

## On-DGX bootstrap (no internet)

```bash
# Once, on the DGX:
cd /path/to/dgx_timeseries_lab
uv sync --offline    # if wheels were pre-cached via UV_CACHE_DIR
# or:
uv pip install --no-index --find-links /path/to/wheelhouse -e .
```

## The synth → parquet workflow for synthetic data

Since the layered generator is deterministic and self-contained, you can generate datasets on any machine and distribute the parquet output:

```powershell
# Generate once (any machine):
uv run dgx-ts synth dataset=presets/leo_eps_24h output_dir=data/synth

# Distribute the output dir (e.g., data/synth/leo_eps_24h/) by sneakernet.

# Load anywhere:
uv run dgx-ts train dataset=parquet model=rolling_mean trainer=single_cpu
# (configs/dataset/parquet.yaml points at data/synth/leo_eps_24h)
```

## MLflow tracking

By default the MLflow logger writes to a local `mlruns/` directory — no network required. If you stand up a self-hosted MLflow server on the DGX, set `MLFLOW_TRACKING_URI=http://localhost:5000` and the logger will use it.

## Verification

After provisioning:

```powershell
uv run pytest packages/ -v
uv run dgx-ts train experiment=phase0_smoke
uv run dgx-ts train experiment=phase1_layered
```

All three should pass. If they don't, file an issue (well — write it down for the next sneakernet round).
