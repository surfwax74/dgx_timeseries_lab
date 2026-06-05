#!/usr/bin/env bash
# Connected-machine helper: download Llama / Mistral GGUF + safetensors for
# sneakernet transfer to an air-gap DGX. Run on a workstation with
# internet, then copy `data/llm_weights/` to the target host.
set -euo pipefail

DEST="${1:-data/llm_weights}"
mkdir -p "$DEST"

# Defaults — override via env if you need others
LLAMA_HF_REPO="${LLAMA_HF_REPO:-meta-llama/Llama-3.1-8B-Instruct}"
MISTRAL_GGUF_URL="${MISTRAL_GGUF_URL:-https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf}"

echo "Destination: $DEST"

# huggingface-cli for safetensors (Llama)
if command -v huggingface-cli >/dev/null 2>&1; then
    echo "Downloading $LLAMA_HF_REPO via huggingface-cli..."
    huggingface-cli download "$LLAMA_HF_REPO" \
        --local-dir "$DEST/$(basename "$LLAMA_HF_REPO")" \
        --local-dir-use-symlinks=False
else
    echo "WARN: huggingface-cli not found; skipping Llama download."
    echo "      pip install huggingface_hub[cli]"
fi

# Direct curl for the Mistral GGUF (smaller, simpler)
if command -v curl >/dev/null 2>&1; then
    echo "Downloading Mistral GGUF..."
    curl -L -o "$DEST/$(basename "$MISTRAL_GGUF_URL")" "$MISTRAL_GGUF_URL"
else
    echo "WARN: curl not found; skipping Mistral download."
fi

echo "Done. Sneakernet-transfer $DEST/ to the air-gap host."
