#!/usr/bin/env bash
# scripts/build_dataset.sh — smart materializer for cached datasets.
#
# What it does:
#   1. Looks for data/synth/<name>/data.parquet
#   2. If present + no --force: prints where it lives, exits 0
#   3. If missing (or --force): resolves the preset name and runs
#      `dgx-ts synth dataset=<resolved>` to materialize it
#   4. On success, prints the parquet path + suggests the next command
#
# Usage:
#     bash scripts/build_dataset.sh <name>                          # build if missing
#     bash scripts/build_dataset.sh <name> --force                  # rebuild regardless
#     bash scripts/build_dataset.sh <name> --preset-group foo       # explicit group
#     bash scripts/build_dataset.sh <name> --output-dir /mnt/nas    # non-default cache dir
#
# Supported <name>s out of the box:
#     trivial_synth       (top-level, no preset group)
#     leo_eps_24h         (via presets/leo_eps_24h)
#     leo_eps_full_24h    (via presets/leo_eps_full_24h)
#
# After materialization, use via `dataset=cached/<name>` in any experiment.
set -euo pipefail

NAME=""
FORCE=0
PRESET_GROUP=""
OUTPUT_DIR="data/synth"

usage() {
    sed -n '2,20p' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)         FORCE=1; shift ;;
        --preset-group)  PRESET_GROUP="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)       usage ;;
        -*)              echo "unknown flag: $1" >&2; exit 2 ;;
        *)               if [[ -z "$NAME" ]]; then NAME="$1"; else echo "extra positional: $1" >&2; exit 2; fi; shift ;;
    esac
done

if [[ -z "$NAME" ]]; then
    echo "ERROR: dataset name is required" >&2
    usage
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
else
    echo "ERROR: no .venv at .venv/bin/python — run 'uv sync' first." >&2
    exit 1
fi
export UV_NO_SYNC=1

CACHE_PATH="$REPO_ROOT/$OUTPUT_DIR/$NAME"
PARQUET_FILE="$CACHE_PATH/data.parquet"

# Cache-hit path — skip unless --force
if [[ -f "$PARQUET_FILE" && $FORCE -eq 0 ]]; then
    echo "[build_dataset] Cache HIT: $CACHE_PATH"
    if [[ -f "$CACHE_PATH/manifest.yaml" ]]; then
        echo "  Manifest:"
        head -n 8 "$CACHE_PATH/manifest.yaml" | sed 's/^/    /'
    fi
    echo
    echo "  Use it via:"
    echo "    dataset=cached/$NAME"
    echo
    echo "  Force rebuild with: bash scripts/build_dataset.sh $NAME --force"
    exit 0
fi

# Cache-miss (or --force) path — resolve the Hydra dataset selector
if [[ -z "$PRESET_GROUP" ]]; then
    if [[ -f "$REPO_ROOT/configs/dataset/presets/$NAME.yaml" ]]; then
        DATASET_SELECTOR="presets/$NAME"
    elif [[ -f "$REPO_ROOT/configs/dataset/$NAME.yaml" ]]; then
        DATASET_SELECTOR="$NAME"
    else
        echo "ERROR: no source YAML found for '$NAME'. Expected one of:" >&2
        echo "  $REPO_ROOT/configs/dataset/presets/$NAME.yaml" >&2
        echo "  $REPO_ROOT/configs/dataset/$NAME.yaml" >&2
        echo "Create one, or pass --preset-group <group> to override." >&2
        exit 1
    fi
else
    DATASET_SELECTOR="$PRESET_GROUP/$NAME"
fi

if [[ $FORCE -eq 1 && -d "$CACHE_PATH" ]]; then
    echo "[build_dataset] --force set: removing existing cache at $CACHE_PATH"
    rm -rf "$CACHE_PATH"
fi

echo "[build_dataset] Cache MISS. Materializing '$NAME' from dataset=$DATASET_SELECTOR..."
echo "  Output dir: $CACHE_PATH"
echo

"$PYTHON" -m dgx_ts_lab.cli.main synth dataset="$DATASET_SELECTOR"

if [[ ! -f "$PARQUET_FILE" ]]; then
    echo "ERROR: synth completed but no parquet at $PARQUET_FILE — check the dataset config's 'name' field matches '$NAME'" >&2
    exit 1
fi

echo
echo "[build_dataset] DONE."
echo "  Cache path: $CACHE_PATH"
echo "  Use it via:"
echo "    dataset=cached/$NAME"
