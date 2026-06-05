#!/usr/bin/env bash
# Boot a vLLM OpenAI-compatible server on the DGX (or any GPU host).
# Designed for air-gap deployment: takes a local model dir, no network use.
#
# Usage:
#   scripts/setup_vllm_server.sh <model_path> [tensor_parallel_size] [port]
#
# Examples:
#   scripts/setup_vllm_server.sh /data/llm_weights/Llama-3.1-70B-Instruct 4 8000
#   scripts/setup_vllm_server.sh /data/llm_weights/Mixtral-8x22B-Instruct 8 8000
set -euo pipefail

MODEL_PATH="${1:-}"
TP_SIZE="${2:-1}"
PORT="${3:-8000}"

if [[ -z "$MODEL_PATH" ]]; then
    echo "usage: $0 <model_path> [tensor_parallel_size] [port]" >&2
    exit 2
fi

if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: 'vllm' not on PATH. Install with: pip install vllm" >&2
    exit 1
fi

echo "Launching vLLM:"
echo "  model:           $MODEL_PATH"
echo "  tensor_parallel: $TP_SIZE"
echo "  port:            $PORT"

exec vllm serve "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
