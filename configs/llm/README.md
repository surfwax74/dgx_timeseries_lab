# `configs/llm/`

Backend configs for the Phase 11 ops co-pilot. All real backends conform
to the same `LLMBackend` Protocol — swap between them by changing one
YAML, no co-pilot code change.

## Picking a config

```
dgx-ts copilot --backend anthropic
dgx-ts copilot --backend vllm --model meta-llama/Llama-3.1-70B-Instruct
dgx-ts copilot --backend ollama --model llama3.1:8b
dgx-ts copilot --backend llama_cpp --model data/llm_weights/mistral-7b-instruct.gguf
dgx-ts copilot --backend mock        # for CI / smoke
```

## Per-tier defaults — recommended pick × hardware

| Tier | Recommended | Why |
|---|---|---|
| Laptop / CI | `mock` | No SDK, no network, deterministic |
| Dev workstation (online) | `anthropic` | Best quality when online is OK |
| RTX 3080 (10 GB) | `ollama_llama8b` or `llama_cpp_mistral7b_q4` | Easy setup, fits Q5 8B / Q4 7B |
| A5000 (24 GB) | `ollama_phi4` or `vllm_granite_8b` | Phi-4 14B INT8 or Granite 8B bf16 |
| H200 (single, 141 GB) | `vllm_llama70b` or `vllm_granite_34b_code` | Frontier 70B or top-tier code model |
| 8x H200 (DGX) | `vllm_mistral_8x22b` | Tensor-parallel MoE across all 8 GPUs |
| Air-gap | any vLLM / Ollama / llama.cpp; **never** `anthropic` | API call would breach air-gap |

## When to pick which model (not just which tier)

| If you need… | Pick | Backend |
|---|---|---|
| Maximum frontier quality, online OK | `claude-sonnet-4-5` | `anthropic` |
| Maximum quality, air-gap | Mixtral 8×22B | `vllm_mistral_8x22b` |
| Strong agentic reasoning, single-H200 footprint | Llama 70B INT8 | `vllm_llama70b` |
| **Reliable JSON-mode** for procedure synthesis | **Granite-Code 34B** | `vllm_granite_34b_code` |
| Apache-2.0 licensing for DoD / classified contexts | Granite 3.2 (8B) or Granite-Code | `vllm_granite_8b` / `*_code` |
| Workstation-tier agentic on A5000 | Phi-4 14B | `vllm_phi4` / `ollama_phi4` |
| Small-but-capable on RTX 3080 (10 GB) | Mistral 7B Q4 | `llama_cpp_mistral7b_q4` |
| Pure CI smoke (no SDK, deterministic) | mock backend | `mock` |

## Full config inventory

### Anthropic (online, hosted API)
- `anthropic.yaml` → `claude-sonnet-4-5`

### Meta / Mistral (Llama Community License or Apache 2.0)
- `vllm_llama70b.yaml` → `meta-llama/Llama-3.1-70B-Instruct` (Llama license ⚠)
- `ollama_llama8b.yaml` → `llama3.1:8b` (Llama license ⚠)
- `vllm_mistral_8x22b.yaml` → `mistralai/Mixtral-8x22B-Instruct-v0.1` (Apache 2.0)
- `llama_cpp_mistral7b_q4.yaml` → `mistral-7b-instruct-v0.3.Q4_K_M.gguf` (Apache 2.0)

### IBM Granite (all Apache 2.0)
- `vllm_granite_8b.yaml` → `ibm-granite/granite-3.2-8b-instruct` (general instruct)
- `ollama_granite_8b.yaml` → `granite3.2:8b` (general instruct via Ollama)
- `vllm_granite_8b_code.yaml` → `ibm-granite/granite-8b-code-instruct` (code-specialized)
- `vllm_granite_34b_code.yaml` → `ibm-granite/granite-34b-code-instruct` (top-tier code)

### Microsoft Phi (MIT)
- `vllm_phi4.yaml` → `microsoft/phi-4` (14B)
- `ollama_phi4.yaml` → `phi4:14b` (via Ollama)

## Licensing notes for security review

| License | Models in inventory | Air-gap implication |
|---|---|---|
| Apache 2.0 | Mixtral, Mistral 7B, all Granite, Phi-4 (MIT, equivalent) | Preferred for any deployment |
| MIT | Phi-4 | Preferred |
| Llama 3.1 Community | Llama 3.1 70B, Llama 3.1 8B | Has acceptable-use restrictions and 700M-MAU clause; some government / defense applications may need separate negotiation. Granite is the drop-in Apache replacement. |
| Anthropic API (commercial) | Claude Sonnet 4.5 | NOT air-gap deployable; never use in classified context |

See `docs/foundation_model_roadmap.md` § "Why Granite was added" for the
full Llama-vs-Granite reasoning.
