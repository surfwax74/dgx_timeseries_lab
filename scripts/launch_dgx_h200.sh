#!/usr/bin/env bash
# launch_dgx_h200.sh — DGX 8×H200, the flex tier.
# Pre-req: scripts/install_cuda_torch.sh 128 + export UV_NO_SYNC=1
#          MLflow Registry server up (scripts/setup_mlflow_registry.sh) if you want
#          to use foundation models via models:// URIs.
#
# Auto-detects 1×H200 vs 8×H200 and picks the appropriate trainer config.

set -euo pipefail
EXPERIMENT="${1:-phase4_scale}"
EPOCHS="${2:-100}"

if [ -z "${UV_NO_SYNC:-}" ]; then
    echo "WARNING: UV_NO_SYNC is not set." >&2
fi

echo "==> GPU preflight (require 1×H200 minimum, 8 for full flex)"
uv run --no-sync python scripts/check_gpu.py --require=1 --min-vram-gb=80

N_GPU=$(uv run --no-sync python -c "import torch; print(torch.cuda.device_count())")

# NCCL tuning for H200 fabric
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0

if [ "$N_GPU" -ge 4 ]; then
    TRAINER="h200_fsdp_8x"
    echo "==> Detected $N_GPU H200s → using trainer=h200_fsdp_8x (FSDP, full flex)"
else
    TRAINER="h200"
    echo "==> Single H200 → using trainer=h200"
fi

echo
echo "==> Synth dataset if missing"
if [ ! -f "data/synth/leo_eps_full_24h/data.parquet" ]; then
    uv run --no-sync dgx-ts synth dataset=presets/leo_eps_full_24h
fi

echo
echo "==> Run experiment=$EXPERIMENT on DGX (max_epochs=$EPOCHS, trainer=$TRAINER)"
uv run --no-sync dgx-ts benchmark \
    experiment="$EXPERIMENT" \
    trainer="$TRAINER" \
    trainer.max_epochs="$EPOCHS"
