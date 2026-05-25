#!/usr/bin/env bash
# launch_a5000.sh — A5000 server (1 or N GPUs).
# Pre-req: scripts/install_cuda_torch.sh has been run + export UV_NO_SYNC=1
#
# Auto-detects single vs multi-GPU and picks the appropriate trainer config.

set -euo pipefail
EXPERIMENT="${1:-phase2_bakeoff}"
EPOCHS="${2:-20}"

if [ -z "${UV_NO_SYNC:-}" ]; then
    echo "WARNING: UV_NO_SYNC is not set. uv run will re-sync and revert torch to CPU." >&2
    echo "  export UV_NO_SYNC=1   (or run install_cuda_torch.sh first)" >&2
fi

echo "==> GPU preflight"
uv run --no-sync python scripts/check_gpu.py --require=1 --min-vram-gb=20

N_GPU=$(uv run --no-sync python -c "import torch; print(torch.cuda.device_count())")
if [ "$N_GPU" -ge 4 ]; then
    TRAINER="a5000_x8"
    echo "==> Detected $N_GPU GPUs → using trainer=a5000_x8 (DDP/FSDP)"
else
    TRAINER="a5000"
    echo "==> Single A5000 → using trainer=a5000"
fi

echo
echo "==> Synth dataset if missing"
if [ ! -f "data/synth/leo_eps_full_24h/data.parquet" ]; then
    uv run --no-sync dgx-ts synth dataset=presets/leo_eps_full_24h
fi

echo
echo "==> Run experiment=$EXPERIMENT on A5000 (max_epochs=$EPOCHS, trainer=$TRAINER)"
uv run --no-sync dgx-ts benchmark \
    experiment="$EXPERIMENT" \
    trainer="$TRAINER" \
    trainer.max_epochs="$EPOCHS"
