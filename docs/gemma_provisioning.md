# Google Gemma OSS Provisioning

Deployment path for hosting Google's open-weights Gemma family behind
the `LLMBackend` Protocol so the Phase 11 ops co-pilot can use it. Five
configs ship covering three self-hostable backends (vLLM, Ollama,
llama-cpp) and four sizes (27B / 12B / 4B / 4B-quantized).

> **Version note (2026)**: The configs pin `gemma-3-*` model IDs — that
> is the latest OSS release my authoring context is confident in
> (Gemma 3, March 2025). If Gemma 4 has since released, update the
> `model_id` (vLLM / Ollama) or the `model_path` (llama.cpp) to the
> current variant. **Only the ID string changes** — the backend
> plumbing is identical.

## License — read this first

Gemma ships under the **Gemma Terms of Use** (a Google custom license),
NOT Apache 2.0. Key implications:

- ✅ Commercial use permitted
- ✅ Fine-tuning / redistribution permitted (with attribution + license copy)
- ❗ Subject to Google's Prohibited Use Policy (surveillance, weapons,
  disinformation, CSAM, biometric ID without consent, etc.)
- ❗ Redistribution obligations: your air-gap distribution bundle must
  include the license text and attribution
- ❗ Not on the "Apache-2.0-only" allowed list for some DoD / intel
  contracts — verify with your legal team before deployment

**When Apache-2.0 licensing is a hard requirement**, prefer IBM Granite
3.2 (which ships in this repo as `vllm_granite_8b`) — the ops co-pilot
performs comparably on our RAG + telemetry-tool workloads.

## Sizing + tier matrix

| Config | Params | FP16 VRAM | Target hardware | Backend |
|---|---:|---:|---|---|
| `llama_cpp_gemma_4b_q4.yaml` |  4 B | ~2.5 GB (Q4) | CPU / RTX 3080 (offload) / air-gap | llama.cpp |
| `ollama_gemma_4b.yaml`       |  4 B | ~9 GB        | RTX 3080 (10 GB) / small workstation | Ollama |
| `ollama_gemma_12b.yaml`      | 12 B | ~24 GB       | A5000 / single H200 | Ollama |
| `vllm_gemma_12b.yaml`        | 12 B | ~24 GB       | A5000 / single H200 (throughput) | vLLM |
| `vllm_gemma_27b.yaml`        | 27 B | ~54 GB       | Single H200 (141 GB) | vLLM |

Gemma also supports up to 128k context on the larger sizes — override
`n_ctx` in the llama-cpp config or vLLM's launch flags as needed.

## Provisioning flow — three paths

### Path 1: HuggingFace Hub (connected DGX / server)

```bash
# One-time login
huggingface-cli login                          # accept the Gemma license first at
                                               # https://huggingface.co/google/gemma-3-27b-it

# Sneakernet-friendly download to a known path
huggingface-cli download google/gemma-3-27b-it \
    --local-dir /data/llm_weights/gemma-3-27b-it \
    --local-dir-use-symlinks False

# Serve via vLLM
bash scripts/setup_vllm_server.sh /data/llm_weights/gemma-3-27b-it 1 8000

# Point the co-pilot at it
dgx-ts copilot --backend vllm \
    --model google/gemma-3-27b-it \
    --base-url http://localhost:8000/v1
```

### Path 2: Ollama (fastest local dev-workstation setup)

```bash
# One command per size
ollama pull gemma3:27b        # or gemma3:12b / gemma3:4b / gemma3:1b
ollama serve &

# Use it
dgx-ts copilot --backend ollama --model gemma3:27b
```

Ollama automatically applies quantization (default: Q4_K_M) to fit
consumer VRAM. Great for iteration; use vLLM for production throughput.

### Path 3: Air-gapped DGX (sneakernet)

Follow Path 1 on a connected staging box, then:

```bash
# On the connected box
tar czf gemma-3-27b-it.tar.gz -C /data/llm_weights gemma-3-27b-it
sha256sum gemma-3-27b-it.tar.gz > gemma-3-27b-it.tar.gz.sha256

# Transfer both files to the DGX

# On the DGX
sha256sum -c gemma-3-27b-it.tar.gz.sha256
tar xzf gemma-3-27b-it.tar.gz -C /data/llm_weights
bash scripts/setup_vllm_server.sh /data/llm_weights/gemma-3-27b-it 1 8000
```

For the CPU-tier `llama_cpp_gemma_4b_q4.yaml`, ship a single GGUF file
instead (a `.tar.gz` is unnecessary). Grab the GGUF from:

- Official conversions: https://huggingface.co/google (may or may not
  ship official GGUF — check the release card)
- Community GGUFs (verify checksums before ingesting into a secure env):
  - https://huggingface.co/bartowski
  - https://huggingface.co/lmstudio-community

## Verifying a deployment

```bash
# Health check the vLLM server
curl http://localhost:8000/v1/models

# Live smoke via the co-pilot mock harness
dgx-ts copilot --backend vllm \
    --model google/gemma-3-27b-it \
    --base-url http://localhost:8000/v1 \
    --prompt "One sentence: what does a battery-SoC anomaly typically indicate?"
```

Expected: a short domain-appropriate answer within ~2 s (H200) / ~8 s
(A5000) / ~30 s (CPU llama-cpp).

## Updating when Gemma 4 (or 5, etc.) lands

The whole point of the `LLMBackend` Protocol is that model swaps are
one-line changes. When a new version drops:

1. Verify the HuggingFace model ID at https://huggingface.co/google.
2. Update `model_id` in each `configs/llm/vllm_gemma_*.yaml` and
   `ollama_gemma_*.yaml` (or the `model_path` GGUF filename in the
   llama-cpp config).
3. Update the Ollama tag (`gemma4:12b` etc.) if the Ollama library
   uses a different naming scheme.
4. Optionally rename the YAML files themselves for clarity
   (`vllm_gemma_27b.yaml` → `vllm_gemma4_27b.yaml`) — the co-pilot
   accepts the config path via `--config`, so it's cosmetic.
5. Re-run the smoke verification above.
6. Commit the bump and note the release date in the commit message.

The `LLMBackend` code layer never touches the model ID directly — every
backend reads it from its config file — so there is no source change
required at any point.

## Cross-references

- Backend protocol + all 4 backends: `packages/dgx_ts_lab/src/dgx_ts_lab/llm/README.md`
- Full LLM inventory + tier picker: `configs/llm/README.md`
- Foundation model roadmap (why some LLMs are in the roadmap, some not):
  `docs/foundation_model_roadmap.md`
- Cookbook Phase 11 (co-pilot recipes): `docs/experiments_cookbook.md`
