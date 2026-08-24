#!/usr/bin/env bash
# Boot a vLLM OpenAI-compatible server on the DGX (or any GPU host).
# Designed for air-gap deployment: takes a local model dir, no network use.
#
# Usage:
#   scripts/setup_vllm_server.sh <model_path> [tensor_parallel_size] [port] [tool_parser]
#
# Examples:
#   scripts/setup_vllm_server.sh /data/llm_weights/Llama-3.1-70B-Instruct 4 8000
#   scripts/setup_vllm_server.sh /data/llm_weights/Mixtral-8x22B-Instruct 8 8000 mistral
#   scripts/setup_vllm_server.sh /data/llm_weights/gemma-4-26B-A4B-it 1 8000 gemma
#
# TOOL PARSER (4th arg) — why it matters:
#   The Phase 11 co-pilot calls telemetry tools through the OpenAI
#   tool-calling API. vLLM needs a model-family-appropriate parser to
#   turn the model's raw output back into structured tool calls. Passing
#   the wrong parser does NOT error loudly — tool calls just silently
#   fail to parse and the co-pilot degrades to plain chat. Match the
#   parser to the model family:
#
#     llama3_json   Llama 3.x            (default, back-compat)
#     mistral       Mistral / Mixtral
#     gemma         Gemma 3 / Gemma 4
#     granite       IBM Granite
#     hermes        Qwen, Phi, many others using Hermes-style calls
#
#   Check `vllm serve --help | grep -A5 tool-call-parser` for the list
#   your installed vLLM version actually supports — names drift between
#   releases.
#
# EXTRA ARGS:
#   Set VLLM_EXTRA_ARGS to append anything else, e.g. long context:
#     VLLM_EXTRA_ARGS="--max-model-len 32768 --gpu-memory-utilization 0.90" \
#       scripts/setup_vllm_server.sh /data/llm_weights/gemma-4-26B-A4B-it 1 8000 gemma
set -euo pipefail

MODEL_PATH="${1:-}"
TP_SIZE="${2:-1}"
PORT="${3:-8000}"
TOOL_PARSER="${4:-llama3_json}"

if [[ -z "$MODEL_PATH" ]]; then
    echo "usage: $0 <model_path> [tensor_parallel_size] [port] [tool_parser]" >&2
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
echo "  tool_parser:     $TOOL_PARSER"
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    echo "  extra_args:      $VLLM_EXTRA_ARGS"
fi

# shellcheck disable=SC2086  # VLLM_EXTRA_ARGS is intentionally word-split
exec vllm serve "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    ${VLLM_EXTRA_ARGS:-}
