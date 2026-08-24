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
| RTX 3080 (10 GB) | `ollama_gemma4_e4b` or `llama_cpp_mistral7b_q4` | Gemma 4 E4B (~9.6 GB, Apache 2.0) or Q4 Mistral 7B |
| A5000 (24 GB) | `vllm_gemma4_12b` / `ollama_phi4` / `vllm_granite_8b` | Gemma 4 12B (256K ctx + audio), Phi-4 14B, Granite 8B |
| H200 (single, 141 GB) | **`vllm_gemma4_26b_a4b`** / `vllm_llama33_70b` (FP8) / `vllm_gemma4_31b` | MoE at ~14 GB leaves the card free; Llama 3.3 70B FP8 (~71 GB) for max quality on one card |
| 8x H200 (DGX) — production | **`vllm_gemma4_26b_a4b`** | ~14 GB for the co-pilot leaves ~7.9 GPUs for training. Displaces `vllm_mistral_8x22b`, which needs multi-GPU sharding. |
| 8x H200 (DGX) — **ceiling / stress** | `vllm_llama31_405b_bf16` | ~810 GB at TP=8. Fits H200 (1128 GB); **does not fit** 8x H100 / 8x A100 (640 GB). Consumes the whole box. |
| Air-gap | any vLLM / Ollama / llama.cpp; **never** `anthropic` | API call would breach air-gap |

## When to pick which model (not just which tier)

| If you need… | Pick | Backend |
|---|---|---|
| Maximum frontier quality, online OK | `claude-sonnet-4-5` | `anthropic` |
| **Maximum quality, air-gap, cost no object** | **Llama 3.1 405B bf16** | `vllm_llama31_405b_bf16` |
| 405B quality with usable throughput + context | Llama 3.1 405B FP8 | `vllm_llama31_405b_fp8` |
| Near-frontier while leaving the box free | Llama 3.3 70B (FP8) | `vllm_llama33_70b` |
| Strong agentic reasoning, single-H200 footprint | Llama 3.3 70B | `vllm_llama33_70b` (supersedes `vllm_llama70b`) |
| **Reliable JSON-mode** for procedure synthesis | **Granite-Code 34B** | `vllm_granite_34b_code` |
| Apache-2.0 licensing for DoD / classified contexts | Granite 3.2 (8B) or Granite-Code | `vllm_granite_8b` / `*_code` |
| Workstation-tier agentic on A5000 | Phi-4 14B | `vllm_phi4` / `ollama_phi4` |
| **Best capability per VRAM byte** (co-pilot default) | **Gemma 4 26B-A4B MoE** | `vllm_gemma4_26b_a4b` |
| Max open-model reasoning, single H200 | Gemma 4 31B dense | `vllm_gemma4_31b` |
| Audio input (future voice ops console) | Gemma 4 12B / E4B / E2B | `vllm_gemma4_12b` / `ollama_gemma4_e4b` |
| 256K-context RAG over procedures | Gemma 4 (12B and up) | `vllm_gemma4_12b` / `*_26b_a4b` / `*_31b` |
| Small-but-capable on RTX 3080 (10 GB) | Gemma 4 E4B or Mistral 7B Q4 | `ollama_gemma4_e4b` / `llama_cpp_mistral7b_q4` |
| Air-gap, no GPU | Gemma 4 E2B Q4 GGUF | `llama_cpp_gemma4_e2b_q4` |
| Pure CI smoke (no SDK, deterministic) | mock backend | `mock` |

## Full config inventory

### Anthropic (online, hosted API)
- `anthropic.yaml` → `claude-sonnet-4-5`

### Meta / Mistral (Llama Community License or Apache 2.0)
- `vllm_llama31_405b_bf16.yaml` → `meta-llama/Llama-3.1-405B-Instruct` — **~810 GB, TP=8. H200-only capability cliff.** (Llama license ⚠)
- `vllm_llama31_405b_fp8.yaml` → `meta-llama/Llama-3.1-405B-Instruct-FP8` (~405 GB, TP=8) (Llama license ⚠)
- `vllm_llama33_70b.yaml` → `meta-llama/Llama-3.3-70B-Instruct` — **preferred 70B**; supersedes 3.1 70B (Llama license ⚠)
- `vllm_llama70b.yaml` → `meta-llama/Llama-3.1-70B-Instruct` (legacy; prefer 3.3) (Llama license ⚠)
- `ollama_llama8b.yaml` → `llama3.1:8b` (Llama license ⚠)
- `vllm_mistral_8x22b.yaml` → `mistralai/Mixtral-8x22B-Instruct-v0.1` (Apache 2.0)
- `llama_cpp_mistral7b_q4.yaml` → `mistral-7b-instruct-v0.3.Q4_K_M.gguf` (Apache 2.0)

> **Sizing + the H200 capability cliff** for the 70B / 405B tier:
> [`docs/frontier_model_serving.md`](../../docs/frontier_model_serving.md).
> Short version: 405B at bf16 needs ~810 GB, which fits 8× H200
> (1128 GB) and does **not** fit 8× H100 or 8× A100 (640 GB).

### IBM Granite (all Apache 2.0)
- `vllm_granite_8b.yaml` → `ibm-granite/granite-3.2-8b-instruct` (general instruct)
- `ollama_granite_8b.yaml` → `granite3.2:8b` (general instruct via Ollama)
- `vllm_granite_8b_code.yaml` → `ibm-granite/granite-8b-code-instruct` (code-specialized)
- `vllm_granite_34b_code.yaml` → `ibm-granite/granite-34b-code-instruct` (top-tier code)

### Microsoft Phi (MIT)
- `vllm_phi4.yaml` → `microsoft/phi-4` (14B)
- `ollama_phi4.yaml` → `phi4:14b` (via Ollama)

### Google Gemma 4 (Apache 2.0, released 2 Apr 2026)
- `vllm_gemma4_26b_a4b.yaml` → `google/gemma-4-26B-A4B-it` — **MoE, 3.8B active, ~14 GB. Recommended DGX default.**
- `vllm_gemma4_31b.yaml` → `google/gemma-4-31B-it` (30.7B dense, single H200)
- `vllm_gemma4_12b.yaml` → `google/gemma-4-12B-it` (A5000; 256K ctx + audio in)
- `ollama_gemma4_12b.yaml` → `gemma4:12b` (workstation via Ollama)
- `ollama_gemma4_e4b.yaml` → `gemma4:e4b` (RTX 3080; Per-Layer Embeddings)
- `llama_cpp_gemma4_e2b_q4.yaml` → `gemma-4-E2B-it.Q4_K_M.gguf` (air-gap CPU)

> ⚠ **Serving gotcha**: pass `gemma` as the 4th arg to
> `scripts/setup_vllm_server.sh` so vLLM picks the right tool-call parser.
> The default (`llama3_json`) fails *silently* — chat works, telemetry
> tool calls quietly stop parsing. Details:
> [`docs/gemma_provisioning.md`](../../docs/gemma_provisioning.md).

## Licensing notes for security review

| License | Models in inventory | Air-gap implication |
|---|---|---|
| Apache 2.0 | **Gemma 4 (all sizes)**, Mixtral, Mistral 7B, all Granite | Preferred for any deployment, including classified |
| MIT | Phi-4 | Preferred |
| Llama 3.1 / 3.3 Community | Llama 3.1 405B, Llama 3.3 70B, Llama 3.1 70B, Llama 3.1 8B | Acceptable-use restrictions + 700M-MAU clause. Repos are **gated** — someone must accept the license on HF before download, which adds a step to the air-gap staging flow. Some government / defense applications may need separate negotiation. Gemma 4 or Granite are the drop-in Apache replacements. |
| Anthropic API (commercial) | Claude Sonnet 4.5 | NOT air-gap deployable; never use in classified context |

> **Changed in Gemma 4**: Gemma 3 shipped under a custom "Gemma Terms of
> Use" carrying a Prohibited Use Policy and redistribution obligations,
> which kept it off the Apache-only allow-list some DoD contracts
> require. Gemma 4 is plain Apache 2.0 — it now sits in the same
> licensing tier as Granite, with materially better capability per VRAM
> byte. The Gemma 3 configs were removed in favour of Gemma 4; recover
> them from git history if you already have Gemma 3 weights staged.

See `docs/foundation_model_roadmap.md` § "Why Granite was added" for the
full Llama-vs-Granite reasoning. Provisioning walkthrough for the
Gemma 4 family: [`docs/gemma_provisioning.md`](../../docs/gemma_provisioning.md).
