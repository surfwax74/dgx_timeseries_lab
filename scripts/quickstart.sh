#!/usr/bin/env bash
# scripts/quickstart.sh — one-command demo on Linux / DGX.
#
# What this does:
#   1. Verify the venv exists (creates one via `uv sync` if missing)
#   2. Run a tiny bake-off (3 detectors x 1 synth dataset x 2 seeds) on CPU
#   3. Render presentation-grade ROC / PR / AUC plots from the result
#   4. Print where everything landed
#
# Usage (from repo root):
#     bash scripts/quickstart.sh
#     bash scripts/quickstart.sh --skip-benchmark   # plots only, reuse last run
set -euo pipefail

SKIP_BENCH=0
OUTPUT_DIR="benchmark_reports/quickstart_viz"
FIGURES_DIR="benchmark_reports/quickstart_viz/figures"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-benchmark) SKIP_BENCH=1; shift ;;
        --output-dir)     OUTPUT_DIR="$2"; FIGURES_DIR="$OUTPUT_DIR/figures"; shift 2 ;;
        --figures-dir)    FIGURES_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
echo "[quickstart] repo root: $REPO_ROOT"

# 1. Locate Python — prefer the in-repo .venv, bootstrap with uv otherwise
if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    echo "[quickstart] .venv missing; bootstrapping with uv sync..."
    uv sync
    PYTHON=".venv/bin/python"
else
    echo "ERROR: no .venv and no 'uv' on PATH." >&2
    echo "       Install uv: https://docs.astral.sh/uv/" >&2
    exit 1
fi
echo "[quickstart] python: $PYTHON"

export UV_NO_SYNC=1

# 2. Bake-off
if [[ $SKIP_BENCH -eq 0 ]]; then
    echo
    echo "[quickstart] Step 1/2: running bake-off (experiment=quickstart_viz)..."
    "$PYTHON" -m dgx_ts_lab.cli.main benchmark experiment=quickstart_viz
else
    echo "[quickstart] --skip-benchmark: reusing existing $OUTPUT_DIR"
fi

# 3. Figures
echo
echo "[quickstart] Step 2/2: rendering figures..."
"$PYTHON" -m dgx_ts_lab.cli.main viz \
    --benchmark-dir "$OUTPUT_DIR" \
    --output-dir "$FIGURES_DIR" \
    --format png,svg \
    --splits val,test

echo
echo "[quickstart] DONE."
echo "  Benchmark report: $OUTPUT_DIR/benchmark_report.md"
echo "  Figures (PNG/SVG): $FIGURES_DIR"
echo
echo "  Drop the PNGs straight into your slide deck."
