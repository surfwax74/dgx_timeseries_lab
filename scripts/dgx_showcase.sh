#!/usr/bin/env bash
# scripts/dgx_showcase.sh — the procurement-justifying demo.
#
# Designed to run end-to-end on an 8x H200 DGX in ~6-8 hours. Produces:
#   * trained Sat-TSFM XL multi-task foundation model (1.5B params, FSDP)
#   * trained Sat-MultiModal LARGE cross-modal foundation model
#   * exported ONNX + model_card + feature_schema for both
#   * vLLM-served Mixtral 8x22B ops co-pilot
#   * recorded co-pilot Q&A transcript using the trained detector's outputs
#   * MLflow artifact bundle suitable for the procurement deck
#
# Each step prints elapsed wall-clock so you can quote real numbers on
# the slide ("this run took 4h12m; same workload on shared 8x A100 cluster:
# 38h6m queued + run").
#
# Skip individual steps with --skip-{tsfm,multimodal,export,llm,copilot}.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_TSFM=0; SKIP_MULTIMODAL=0; SKIP_EXPORT=0; SKIP_LLM=0; SKIP_COPILOT=0
MIXTRAL_WEIGHTS="${MIXTRAL_WEIGHTS:-/data/llm_weights/Mixtral-8x22B-Instruct-v0.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/dgx_showcase}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-tsfm)       SKIP_TSFM=1;       shift ;;
        --skip-multimodal) SKIP_MULTIMODAL=1; shift ;;
        --skip-export)     SKIP_EXPORT=1;     shift ;;
        --skip-llm)        SKIP_LLM=1;        shift ;;
        --skip-copilot)    SKIP_COPILOT=1;    shift ;;
        --mixtral-weights) MIXTRAL_WEIGHTS="$2"; shift 2 ;;
        --output-root)     OUTPUT_ROOT="$2";  shift 2 ;;
        -h|--help)         sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUTPUT_ROOT"
SUMMARY="$OUTPUT_ROOT/SHOWCASE_SUMMARY.md"
: > "$SUMMARY"

# ── Helpers ──────────────────────────────────────────────────────────
PY=".venv/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: no venv at $PY (run uv sync first)" >&2; exit 1; }

log_section() { echo; echo "════════════════════════════════════════════════════════════════"
                echo " $*"
                echo "════════════════════════════════════════════════════════════════"; }

run_timed() {
    local label="$1"; shift
    log_section "$label"
    local t0=$(date +%s)
    "$@" 2>&1 | tee -a "$OUTPUT_ROOT/$label.log"
    local rc=${PIPESTATUS[0]}
    local elapsed=$(( $(date +%s) - t0 ))
    local h=$(( elapsed / 3600 )); local m=$(( (elapsed % 3600) / 60 )); local s=$(( elapsed % 60 ))
    printf "  → %s: rc=%d  elapsed=%dh%02dm%02ds\n" "$label" "$rc" "$h" "$m" "$s"
    printf "| %-30s | %d | %dh%02dm%02ds |\n" "$label" "$rc" "$h" "$m" "$s" >> "$SUMMARY.tmp"
    return "$rc"
}

cat > "$SUMMARY" <<EOF
# DGX 8x H200 Showcase Run

**Started:** $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Output:** \`$OUTPUT_ROOT\`
**Hardware:** $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
**Driver / CUDA:** $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) / $(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)

## Steps

| Step | rc | Wall-clock |
|---|---:|---:|
EOF
: > "$SUMMARY.tmp"

# ── Step 1: Pretrain Sat-TSFM XL with multi-task heads (FSDP) ────────
if [[ $SKIP_TSFM -eq 0 ]]; then
    run_timed "01_pretrain_sat_tsfm_xl" \
        "$PY" -m dgx_ts_lab.cli.main train experiment=dgx_showcase
fi

# ── Step 2: Pretrain Sat-MultiModal LARGE (cross-modal MAE, FSDP) ────
if [[ $SKIP_MULTIMODAL -eq 0 ]]; then
    run_timed "02_pretrain_sat_multimodal_large" \
        "$PY" -m dgx_ts_lab.cli.main train experiment=dgx_showcase_multimodal
fi

# ── Step 3: Export both models as MLOps lift artifacts ───────────────
if [[ $SKIP_EXPORT -eq 0 ]]; then
    run_timed "03_export_sat_tsfm_xl" \
        "$PY" -m dgx_ts_lab.cli.main export \
            --checkpoint "checkpoints/sat_tsfm_xl_multitask_fsdp_8xh200/last.ckpt" \
            --format onnx \
            --output "$OUTPUT_ROOT/exports/sat_tsfm_xl"
    run_timed "03b_export_sat_multimodal" \
        "$PY" -m dgx_ts_lab.cli.main export \
            --checkpoint "checkpoints/sat_multimodal_large_fsdp_8xh200/last.ckpt" \
            --format onnx \
            --output "$OUTPUT_ROOT/exports/sat_multimodal_large"
fi

# ── Step 4: Spin up vLLM serving Mixtral 8x22B (8-way tensor parallel) ──
VLLM_PID=""
if [[ $SKIP_LLM -eq 0 ]]; then
    log_section "04_launch_vllm_mixtral_8x22b"
    if [[ ! -d "$MIXTRAL_WEIGHTS" ]]; then
        echo "WARN: Mixtral weights not at $MIXTRAL_WEIGHTS — skipping LLM step." | tee -a "$SUMMARY.tmp"
    else
        nohup bash scripts/setup_vllm_server.sh "$MIXTRAL_WEIGHTS" 8 8000 \
            > "$OUTPUT_ROOT/04_vllm.log" 2>&1 &
        VLLM_PID=$!
        echo "  → vLLM launched PID=$VLLM_PID; waiting for /health..."
        for i in {1..120}; do
            if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
                echo "  → vLLM ready after ${i}s"
                break
            fi
            sleep 1
        done
    fi
fi

# ── Step 5: Run a recorded co-pilot Q&A session against trained outputs ──
if [[ $SKIP_COPILOT -eq 0 && -n "$VLLM_PID" ]]; then
    run_timed "05_copilot_demo_qna" \
        "$PY" scripts/dgx_showcase_copilot_qna.py \
            --backend vllm \
            --model mistralai/Mixtral-8x22B-Instruct-v0.1 \
            --base-url http://localhost:8000/v1 \
            --model-card "$OUTPUT_ROOT/exports/sat_tsfm_xl/model_card.yaml" \
            --output "$OUTPUT_ROOT/copilot_transcript.md"
fi

# ── Teardown ─────────────────────────────────────────────────────────
if [[ -n "$VLLM_PID" ]]; then
    log_section "teardown_vllm"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
fi

# ── Render the summary ───────────────────────────────────────────────
cat "$SUMMARY.tmp" >> "$SUMMARY"
rm -f "$SUMMARY.tmp"
echo >> "$SUMMARY"
echo "**Finished:** $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"

log_section "DONE"
echo "  Summary: $SUMMARY"
echo "  Co-pilot transcript: $OUTPUT_ROOT/copilot_transcript.md"
echo "  ONNX exports: $OUTPUT_ROOT/exports/"
