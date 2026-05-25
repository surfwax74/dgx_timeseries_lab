#!/usr/bin/env bash
# install_cuda_torch.sh — install CUDA-enabled PyTorch on Linux.
#
# Run AFTER `uv sync` to swap CPU torch for the CUDA wheel.
# Set UV_NO_SYNC=1 in your shell so subsequent `uv run` calls don't re-sync.
#
# Usage:
#     ./scripts/install_cuda_torch.sh            # cu126 (default)
#     ./scripts/install_cuda_torch.sh 128        # cu128 for newer drivers / H200

set -euo pipefail
CUDA="${1:-126}"
INDEX_URL="https://download.pytorch.org/whl/cu${CUDA}"

echo "==> Installing torch with CUDA ${CUDA} from ${INDEX_URL}"
uv pip install --reinstall torch --index-url "${INDEX_URL}"

echo "==> Verifying CUDA..."
uv run --no-sync python scripts/check_gpu.py

echo
echo "TIP: export UV_NO_SYNC=1 in this shell so 'uv run' skips its sync pass"
echo "    and keeps the CUDA wheel installed."
