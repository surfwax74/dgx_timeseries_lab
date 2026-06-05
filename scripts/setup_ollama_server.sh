#!/usr/bin/env bash
# Boot Ollama on a workstation / server and pre-pull a model.
#
# Usage:
#   scripts/setup_ollama_server.sh [model_tag]
#
# Examples:
#   scripts/setup_ollama_server.sh llama3.1:8b
#   scripts/setup_ollama_server.sh mistral:7b-instruct-q5_K_M
set -euo pipefail

MODEL_TAG="${1:-llama3.1:8b}"

if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: 'ollama' not on PATH." >&2
    echo "Install from https://ollama.com (or your air-gap mirror)" >&2
    exit 1
fi

# Start daemon in background if not already running
if ! pgrep -x "ollama" >/dev/null 2>&1; then
    echo "Starting Ollama daemon..."
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    sleep 2
fi

echo "Pulling model: $MODEL_TAG"
ollama pull "$MODEL_TAG"
echo "Ready. Test with: dgx-ts copilot --backend ollama --model $MODEL_TAG"
