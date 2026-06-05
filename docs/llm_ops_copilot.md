# LLM Ops Co-Pilot (Phase 11)

The Phase 11 co-pilot lets satellite operators interact with the
anomaly-detection system in natural language. Same code runs against four
LLM backends so you can pick the right one per deployment tier.

## Per-tier backend matrix

| Tier                  | Backend          | Model                              | Why                                |
|-----------------------|------------------|------------------------------------|------------------------------------|
| Laptop / CI           | `mock`           | (none)                             | No SDK / no network / deterministic |
| Connected workstation | `anthropic`      | `claude-sonnet-4-5`                | Best quality when online OK         |
| RTX 3080 dev          | `ollama`         | `llama3.1:8b`                      | Easiest setup; fits 8B in 10 GB     |
| A5000 dev             | `ollama` or vLLM | `llama3.1:8b` or `Mistral-7B`     | Either works                        |
| H200 single           | `vllm`           | `Llama-3.1-70B-Instruct`           | PagedAttention + tool use            |
| 8x H200 (DGX, air-gap)| `vllm`           | `Mixtral-8x22B-Instruct`           | Tensor-parallel across GPUs         |
| Pure CPU air-gap      | `llama_cpp`      | `mistral-7b-instruct.Q4_K_M.gguf`  | No daemon; in-process                |

**Hard rule**: never use `anthropic` in an air-gap deployment. The
backend silently tries to talk to api.anthropic.com.

## What the co-pilot can do

Three capabilities, all backend-agnostic:

1. **Interactive chat (`Copilot`)** — multi-turn ops Q&A with tool use.
   The LLM can call `query_telemetry`, `query_anomaly_history`,
   `lookup_procedure`, `read_model_card`. Tool results are fed back; loop
   continues until the LLM is done (capped at 6 iterations).

2. **Report polish (`ReportGenerator`, B7)** — takes a Phase 7
   explanation Markdown skeleton and asks the LLM to write an Executive
   Summary on top + Recommended Actions on the bottom, citing tool
   results inline. Skeleton is preserved unchanged.

3. **Procedure synthesis (`ProcedureSynthesizer`, B8)** — natural-language
   request → validated command sequence. JSON-only LLM responses, opcode
   vocabulary check, and a user-supplied simulator validator. Retries on
   parse / vocab / simulator errors (capped at 4 iterations by default).

## Setup walkthrough — air-gap DGX (vLLM)

```bash
# (on connected workstation)
scripts/download_local_llm_weights.sh data/llm_weights
# sneakernet `data/llm_weights/` to the DGX

# (on DGX)
scripts/setup_vllm_server.sh /data/llm_weights/Llama-3.1-70B-Instruct 4 8000

# (in another shell on DGX)
dgx-ts copilot --backend vllm \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --procedures docs/procedures/ \
    --model-card runs/last/model_card.yaml
```

## Setup walkthrough — RTX 3080 dev (Ollama)

```bash
scripts/setup_ollama_server.sh llama3.1:8b
dgx-ts copilot --backend ollama --model llama3.1:8b
```

## Setup walkthrough — connected dev (Anthropic)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
dgx-ts copilot --backend anthropic
```

## Tool registry — adding a tool

Edit `packages/dgx_ts_lab/src/dgx_ts_lab/llm/telemetry_tools.py`:

```python
_MY_TOOL_DEF = ToolDef(
    name="my_tool",
    description="What it does",
    parameters={"type": "object", "properties": {"arg1": {"type": "string"}}, "required": ["arg1"]},
)

def _my_tool(args: dict, ctx: CopilotContext) -> str:
    return f"hello {args['arg1']}"

# Then register it in default_tool_registry:
reg.register(_MY_TOOL_DEF, _my_tool)
```

The tool description is sent verbatim to the LLM and is the primary
signal for whether/when the model decides to invoke it. Be concrete and
short — "Returns mean/std/min/max for a channel" beats "queries data".

## Limits

- No streaming UI in `dgx-ts copilot` REPL yet; backends support
  `stream()` but the REPL only uses `generate()`.
- No multi-modal input (image / audio); telemetry tools convert
  everything to text before the LLM sees it.
- No automatic prompt-version tracking across MLflow yet.
- Procedure synthesizer parses JSON only; for richer constraints (DSL
  with conditionals, branching) we'd need a structured-output grammar.
