# Frontier Model Serving — 70B and 405B on the DGX

How to host the two largest models in the inventory, and — more useful
for the procurement conversation — exactly where the hardware boundaries
fall.

## The capability cliff, in bytes

Weight footprint is `params × bytes-per-param`. KV cache, activations,
and allocator fragmentation come on top, so treat the weight number as a
floor, not a budget.

| Model | bf16 | FP8 | INT4 |
|---|---:|---:|---:|
| Llama 3.3 70B | ~141 GB | ~71 GB | ~35 GB |
| Llama 3.1 405B | **~810 GB** | ~405 GB | ~203 GB |

Against single-node capacities:

| Node | Total VRAM | 405B bf16 | 405B FP8 | 70B bf16 |
|---|---:|:---:|:---:|:---:|
| 8× H200 (141 GB) | **1128 GB** | ✅ fits, ~318 GB spare | ✅ comfortable | ✅ TP=2 |
| 8× H100 (80 GB) | 640 GB | ❌ **does not fit** | ✅ ~235 GB spare | ✅ TP=2 |
| 8× A100 (80 GB) | 640 GB | ❌ **does not fit** | ✅ ~235 GB spare | ✅ TP=2 |
| 4× H200 | 564 GB | ❌ does not fit | ✅ tight | ✅ TP=2 |
| 1× H200 | 141 GB | ❌ | ❌ | ⚠ weights only, no KV room |

**The headline**: hosting Llama 3.1 405B at unquantized bf16 in a single
node is H200-exclusive. H100 and A100 nodes of the same GPU count must
quantize to FP8 — they physically cannot hold the model otherwise. This
is not a throughput difference that a longer runtime buys back; it is a
capability that either exists on the box or does not.

Two secondary points that strengthen the same argument:

- **NVSwitch is load-bearing at TP=8.** Tensor-parallel inference
  all-reduces across all eight GPUs on every token, so interconnect
  bandwidth sits on the critical path. NVIDIA measured a further ~1.5×
  throughput gain on 405B specifically from H200 + NVLink Switch. A
  federated pool of PCIe-attached cards degrades badly at this width.
- **Quantization is not free.** FP8 405B scores ~95.4% exact-match on
  GSM8K versus ~96.8% for bf16. Small, but it is the difference between
  "we run the reference model" and "we run an approximation of it" —
  which matters if the co-pilot's output ever needs to be defensible.

## The three-tier serving ladder

These configs serve different purposes; they are not ranked
alternatives.

| Purpose | Config | GPU cost | Box left for training |
|---|---|---:|---|
| **Production co-pilot** | `vllm_gemma4_26b_a4b` | ~14 GB | ~7.9 GPUs |
| **Max quality, still co-resident** | `vllm_llama33_70b` (FP8) | ~71 GB | ~7.5 GPUs |
| **Ceiling / stress / showcase** | `vllm_llama31_405b_bf16` | all 8 GPUs | **nothing** |

Read that table as the dual-use story: routine ops cost a rounding
error, and the same box still has a ceiling nobody else in the building
can reach. Both halves are worth a slide, and they are not in tension —
they are different operating modes of one machine.

`vllm_llama31_405b_fp8` sits between the last two: still all 8 GPUs, but
with enough KV headroom for real batch throughput and long context. It
is the right choice when you actually want to *use* 405B for work rather
than demonstrate that it fits.

## Llama 3.3 70B

A straight upgrade over the `vllm_llama70b.yaml` (Llama 3.1 70B) already
in the inventory — identical parameter count, better instruction
following and reasoning. Meta positions it as approaching 3.1-405B
quality at 70B cost.

```bash
huggingface-cli login    # gated repo: accept the license on the model
                         # page first, or the download 403s

huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
    --local-dir /data/llm_weights/Llama-3.3-70B-Instruct \
    --local-dir-use-symlinks False

# bf16 needs two cards — one 141 GB H200 holds the weights with
# essentially zero room left for KV cache.
scripts/setup_vllm_server.sh /data/llm_weights/Llama-3.3-70B-Instruct 2 8000

dgx-ts copilot --backend vllm \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --base-url http://localhost:8000/v1
```

For the single-GPU FP8 path, pull the FP8 checkpoint instead and pass
`1` for tensor-parallel size. That configuration is the sweet spot if
you want frontier-adjacent quality while leaving the DGX substantially
free.

## Llama 3.1 405B

```bash
# ~810 GB of weights over the wire at bf16. Budget hours, and stage it
# somewhere with room before sneakernetting.
huggingface-cli download meta-llama/Llama-3.1-405B-Instruct \
    --local-dir /data/llm_weights/Llama-3.1-405B-Instruct \
    --local-dir-use-symlinks False

# bf16, all 8 GPUs. Start conservative on context and raise it while
# watching memory — KV cache eats the 318 GB of headroom fast.
VLLM_EXTRA_ARGS="--max-model-len 8192 --gpu-memory-utilization 0.95" \
  scripts/setup_vllm_server.sh /data/llm_weights/Llama-3.1-405B-Instruct 8 8000
```

FP8 instead, when you want throughput rather than a precision claim:

```bash
huggingface-cli download meta-llama/Llama-3.1-405B-Instruct-FP8 \
    --local-dir /data/llm_weights/Llama-3.1-405B-Instruct-FP8 \
    --local-dir-use-symlinks False

VLLM_EXTRA_ARGS="--max-model-len 32768 --gpu-memory-utilization 0.90" \
  scripts/setup_vllm_server.sh /data/llm_weights/Llama-3.1-405B-Instruct-FP8 8 8000
```

If the FP8 repo ID 404s, Meta has renamed these before — try
`meta-llama/Meta-Llama-3.1-405B-Instruct-FP8` or NVIDIA's
`nvidia/Llama-3.1-405B-Instruct-FP8`.

### Expect a slow first start

vLLM loads ~810 GB from disk and shards it across eight GPUs. On NVMe
this is minutes, not seconds; on spinning or network storage it is
considerably worse. The configs set `timeout_s: 600.0` for exactly this
reason. Do not interpret a slow first request as a hang.

## Licensing — read before deploying

Both models ship under the **Llama Community License**, not Apache 2.0:

- Acceptable Use Policy restrictions apply
- 700M-MAU clause (irrelevant at our scale, but legal will ask)
- Gated HF repos — someone must accept the license on the model page
  before `huggingface-cli download` works, which is an extra step in the
  connected-box stage of any air-gap transfer

For DoD, intelligence, or classified deployment where an Apache-only bar
applies, these are the wrong picks regardless of capability. The
licensing-clean alternatives already in the inventory are **Gemma 4**
(Apache 2.0, and the 26B-A4B MoE is excellent) and **IBM Granite**
(Apache 2.0). See the licensing matrix in
[`configs/llm/README.md`](../configs/llm/README.md).

The honest framing for a review board: Llama 3.1 405B is the *capability
demonstration*; Gemma 4 26B-A4B is what you would actually deploy into a
classified enclave.

## Wiring 405B into the procurement showcase

`scripts/dgx_showcase.sh` currently serves Mixtral 8×22B for its co-pilot
step (`--mixtral-weights`). Swapping that to 405B bf16 would make the
showcase's LLM step demonstrate the capability cliff directly rather
than just "we served a big MoE."

That is a deliberate change to a 6–8 hour procurement run, so it is left
as a decision rather than made unilaterally. The tradeoff:

- **Keep Mixtral**: showcase finishes faster, LLM step co-exists with
  other work, story is "we serve a strong open model."
- **Switch to 405B bf16**: LLM step monopolizes the box for its
  duration, but the story becomes "we serve a model that does not fit on
  the alternative hardware you are comparing us against."

`scripts/build_capability_cliff.py` already plots a 405B row in its
dual-use figure, currently at INT4. Updating that to the bf16
fits/does-not-fit comparison above would sharpen figure 4 considerably —
INT4 405B fits nearly anything, so it understates the argument.

## Sources

- [Announcing Llama 3.1 Support in vLLM](https://blog.vllm.ai/2024/07/23/llama31.html)
- [Boosting Llama 3.1 405B Throughput on H200 + NVLink Switch — NVIDIA](https://developer.nvidia.com/blog/boosting-llama-3-1-405b-throughput-by-another-1-5x-on-nvidia-h200-tensor-core-gpus-and-nvlink-switch/)
- [meta-llama/Llama-3.3-70B-Instruct — Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [nvidia/Llama-3.1-405B-Instruct-FP8 — Hugging Face](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP8)
