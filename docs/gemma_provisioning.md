# Google Gemma 4 Provisioning

Deployment path for hosting Google's Gemma 4 open-weights family behind
the `LLMBackend` Protocol so the Phase 11 ops co-pilot can use it. Six
configs ship covering three self-hostable backends (vLLM, Ollama,
llama-cpp) across the full size range.

**Released 2 April 2026 under Apache 2.0.**

## Why this matters for our deployment posture

Two facts about Gemma 4 change the calculus versus every prior open model
in this repo:

**1. Apache 2.0.** Gemma 3 shipped under a custom "Gemma Terms of Use"
with a Prohibited Use Policy and redistribution obligations — enough
friction that we recommended IBM Granite whenever an Apache-only
licensing bar applied. Gemma 4 dropped that entirely. It is now on the
same unencumbered footing as Granite and Mixtral: no acceptable-use
rider, no MAU clause, no attribution burden beyond standard Apache
terms. For DoD / intelligence / classified work, Gemma 4 is
**licensing-clean**.

**2. The 26B-A4B MoE is nearly free to host.** 25.2B total parameters,
but Mixture-of-Experts routing activates only 3.8B per token. In bf16 it
occupies roughly **14 GB — under 10% of one H200's 141 GB** — and serves
at approximately 4B-dense throughput while answering at a much higher
quality tier.

That second point is a direct upgrade to the DGX dual-use capacity
argument. The co-pilot is no longer a workload that competes with
training for whole GPUs; it is a rounding error on a single card. Worth
regenerating `scripts/build_capability_cliff.py`'s `dual_use_capacity`
figure with these numbers before the procurement deck goes out.

## The Gemma 4 family

| Variant | Total | Active/effective | Context | Modality | bf16 VRAM |
|---|---:|---:|---:|---|---:|
| E2B | 5.1B | 2.3B | 128K | text + image + audio | ~5 GB |
| E4B | 8B | 4.5B | 128K | text + image + audio | ~9 GB |
| 12B "Unified" | 11.95B | dense | 256K | text + image + audio | ~24 GB |
| 26B-A4B (MoE) | 25.2B | **3.8B** | 256K | text + image | **~14 GB** |
| 31B Dense | 30.7B | dense | 256K | text + image | ~62 GB |

`E` variants use Per-Layer Embeddings — total weights exceed effective
per-token compute, which is why an "8B" E4B runs closer to a 4.5B model.
Built for on-device and edge deployment.

Audio input lands on E2B / E4B / 12B only. The two largest variants are
text + image. If a voice-driven ops console is ever on the roadmap, 12B
is the largest size that supports it.

## Config inventory

| Config | Variant | Backend | Target |
|---|---|---|---|
| `vllm_gemma4_26b_a4b.yaml` | 26B-A4B MoE | vLLM | **DGX co-pilot default** |
| `vllm_gemma4_31b.yaml` | 31B dense | vLLM | Single H200, max quality |
| `vllm_gemma4_12b.yaml` | 12B | vLLM | A5000, audio-capable |
| `ollama_gemma4_12b.yaml` | 12B | Ollama | Workstation |
| `ollama_gemma4_e4b.yaml` | E4B | Ollama | RTX 3080 / 16 GB |
| `llama_cpp_gemma4_e2b_q4.yaml` | E2B Q4 | llama-cpp | CPU / air-gap edge |

## Which one should I actually run?

**On the DGX, for the ops co-pilot: `vllm_gemma4_26b_a4b`.** It is the
best capability-per-VRAM-byte in the inventory, and its footprint leaves
the box free to train. Reach for `vllm_gemma4_31b` only if you have
measured the MoE falling short on your specific reasoning workload — the
dense 31B is a stronger reasoner but costs ~4.4× the VRAM and serves
notably slower.

**Against the existing inventory**: Gemma 4 26B-A4B should displace
`vllm_mistral_8x22b` as the DGX default. Mixtral 8×22B needs
tensor-parallel sharding across multiple GPUs; Gemma 4 26B-A4B needs
14 GB on one. Benchmark both on your procedure-synthesis and
telemetry-tool workloads before committing, but the resource asymmetry
is stark enough that the burden of proof now sits with Mixtral.

## Provisioning — three paths

### Path 1: HuggingFace Hub (connected host)

```bash
huggingface-cli login       # Apache 2.0 — no license click-through gate

huggingface-cli download google/gemma-4-26B-A4B-it \
    --local-dir /data/llm_weights/gemma-4-26B-A4B-it \
    --local-dir-use-symlinks False

# Serve. Note the 4th arg: the gemma tool-call parser.
scripts/setup_vllm_server.sh /data/llm_weights/gemma-4-26B-A4B-it 1 8000 gemma

dgx-ts copilot --backend vllm \
    --model google/gemma-4-26B-A4B-it \
    --base-url http://localhost:8000/v1
```

Sibling model IDs follow the same shape: `google/gemma-4-31B-it`,
`google/gemma-4-12B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-E2B-it`.
Drop the `-it` suffix for base (non-instruction-tuned) weights — you want
`-it` for the co-pilot.

### Path 2: Ollama (fastest local setup)

```bash
ollama pull gemma4:26b      # MoE — or :31b / :12b / :e4b / :e2b
ollama serve &

dgx-ts copilot --backend ollama --model gemma4:26b
```

Bare `ollama pull gemma4` gets E4B (~9.6 GB), the default tag. Ollama
applies Q4_K_M quantization by default, so on-disk sizes run well below
the bf16 figures in the family table: E2B ~7.2 GB, E4B ~9.6 GB,
26B ~18 GB, 31B ~20 GB.

### Path 3: Air-gapped DGX (sneakernet)

Run Path 1 on a connected staging box, then:

```bash
# On the connected box
tar czf gemma-4-26B-A4B-it.tar.gz -C /data/llm_weights gemma-4-26B-A4B-it
sha256sum gemma-4-26B-A4B-it.tar.gz > gemma-4-26B-A4B-it.tar.gz.sha256

# Transfer both files, then on the DGX:
sha256sum -c gemma-4-26B-A4B-it.tar.gz.sha256
tar xzf gemma-4-26B-A4B-it.tar.gz -C /data/llm_weights
scripts/setup_vllm_server.sh /data/llm_weights/gemma-4-26B-A4B-it 1 8000 gemma
```

Apache 2.0 means no license-acceptance step blocks redistribution into
the enclave — a genuine operational simplification over Gemma 3 and over
anything under the Llama Community License.

For the CPU tier, ship the single E2B GGUF file instead of a tarball.

## Two integration gotchas

**Tool-call parser.** `scripts/setup_vllm_server.sh` takes the parser as
its 4th argument and defaults to `llama3_json` for backward
compatibility. **Pass `gemma` for Gemma models.** Getting this wrong
fails silently — vLLM starts fine, chat works fine, and the co-pilot's
telemetry tool calls simply never parse, so it quietly degrades to
plain conversation. If the co-pilot stops citing actual telemetry, check
this first.

**Context length.** Gemma 4's 256K context is not what vLLM allocates by
default. Request it explicitly:

```bash
VLLM_EXTRA_ARGS="--max-model-len 32768 --gpu-memory-utilization 0.90" \
  scripts/setup_vllm_server.sh /data/llm_weights/gemma-4-26B-A4B-it 1 8000 gemma
```

KV cache grows linearly with context, so 256K on a long-context RAG
workload will consume far more than the ~14 GB weight footprint. Size it
against your actual procedure-corpus length rather than maxing it out.

## Verifying a deployment

```bash
curl http://localhost:8000/v1/models

dgx-ts copilot --backend vllm \
    --model google/gemma-4-26B-A4B-it \
    --base-url http://localhost:8000/v1 \
    --prompt "One sentence: what does a battery-SoC anomaly typically indicate?"
```

Expect a domain-appropriate answer in ~1-2 s on an H200. Then verify
tool calling specifically — that is the part the parser setting breaks:

```bash
dgx-ts copilot --backend vllm \
    --model google/gemma-4-26B-A4B-it \
    --base-url http://localhost:8000/v1 \
    --prompt "Use the telemetry tools to report the current bus voltage."
```

If it answers without invoking a tool, the parser is wrong.

## When the next version lands

Model swaps are one-line changes by design — the `LLMBackend` code layer
never references a model ID directly; every backend reads it from its
config file.

1. Verify current IDs at <https://huggingface.co/google> and
   <https://ai.google.dev/gemma>.
2. Update `model_id` in the `vllm_gemma4_*.yaml` / `ollama_gemma4_*.yaml`
   configs, or `model_path` for llama-cpp.
3. Check whether the vLLM tool-call parser name changed for the new
   family.
4. Re-run both verification commands above, including the tool-call one.
5. Confirm the license still permits your deployment context — do not
   assume it carries forward. Gemma went custom-license → Apache 2.0
   between 3 and 4; it could move again.

## Cross-references

- Backend protocol + all four backends: `packages/dgx_ts_lab/src/dgx_ts_lab/llm/README.md`
- Full LLM inventory + tier picker: [`configs/llm/README.md`](../configs/llm/README.md)
- Co-pilot recipes: [`docs/experiments_cookbook.md`](experiments_cookbook.md) § Phase 11
- Procurement figures that should be regenerated with the MoE numbers:
  `scripts/build_capability_cliff.py`

## Sources

- [Gemma 4 model card — Google AI for Developers](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 announcement — Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [google/gemma-4-26B-A4B-it — Hugging Face](https://huggingface.co/google/gemma-4-26B-A4B-it)
- [Gemma 4 usage guide — vLLM recipes](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html)
- [gemma4 — Ollama library](https://ollama.com/library/gemma4)
