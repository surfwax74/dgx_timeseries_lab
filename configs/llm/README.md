# `configs/llm/`

Backend configs for the Phase 11 ops co-pilot. Pick one with:

```
dgx-ts copilot --backend anthropic
dgx-ts copilot --backend vllm --model meta-llama/Llama-3.1-70B-Instruct
dgx-ts copilot --backend ollama --model llama3.1:8b
dgx-ts copilot --backend llama_cpp --model data/llm_weights/mistral-7b-instruct.gguf
dgx-ts copilot --backend mock        # for CI / smoke
```

## Per-tier defaults

| Tier              | Recommended       | Why                                                  |
|-------------------|-------------------|------------------------------------------------------|
| Laptop / CI       | `mock`            | No SDK / no network / deterministic                  |
| Dev workstation   | `anthropic`       | Best quality when online is OK                       |
| RTX 3080          | `ollama_llama8b`  | Easy setup, fits Q5 8B                               |
| A5000 (single)    | `ollama_llama8b`  | Or vllm for higher throughput                        |
| H200 (single)     | `vllm_llama70b`   | vLLM PagedAttention saves memory                     |
| 8x H200 (DGX)     | `vllm_mistral_8x22b` | Tensor-parallel across GPUs                       |
| Air-gap          | any vLLM / Ollama / llama.cpp; never `anthropic` |    |

All four real backends conform to the same `LLMBackend` Protocol — swap
between them by changing one YAML, no co-pilot code change.
