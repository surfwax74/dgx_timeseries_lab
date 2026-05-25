#!/usr/bin/env bash
# setup_dgx.sh — first-time setup on the DGX (Linux).
#
# Idempotent. Air-gap safe.
# Run from repo root: ./scripts/setup_dgx.sh

set -euo pipefail

echo "==> Checking for uv..."
if ! command -v uv >/dev/null 2>&1; then
    echo "    uv not on PATH."
    echo "    Air-gap: pre-install uv before running this script."
    echo "    Connected: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "    uv: $(uv --version)"

echo "==> Syncing workspace (offline if cache present)..."
if [ -d "${UV_CACHE_DIR:-$HOME/.cache/uv}" ]; then
    uv sync --offline || uv sync
else
    uv sync
fi

echo "==> Creating data/ subdirs (gitignored)..."
mkdir -p data/nasa_smap/{train,test}
mkdir -p data/nasa_msl/{train,test}
mkdir -p data/synth
mkdir -p mlruns

echo "==> Running smoke tests..."
uv run pytest packages/ -q

echo "==> Done."
echo
echo "Next:"
echo "  - Provision NASA data into data/nasa_smap/ and data/nasa_msl/ (see docs/air_gapped_setup.md)"
echo "  - uv run dgx-ts train experiment=phase0_smoke"
echo "  - uv run dgx-ts train experiment=phase1_layered"
