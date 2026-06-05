# `dgx_ts_lab.llm`

Phase 11 — LLM ops co-pilot. All callers (`copilot.py`, `report_generator.py`,
`procedure_synth.py`) talk to a single `LLMBackend` Protocol. Four
implementations live here, plus a `MockBackend` for tests.

## Why four backends

| Backend          | When to use                                   | SDK / process                 |
|------------------|------------------------------------------------|-------------------------------|
| `AnthropicBackend` | Online dev workstations; best quality          | `anthropic` SDK + API key      |
| `VLLMBackend`      | Production DGX (8x H200), throughput-bound     | vLLM HTTP server (OpenAI API)  |
| `OllamaBackend`    | RTX 3080 / single-A5000 dev, easy setup        | Ollama daemon                  |
| `LlamaCppBackend`  | CPU laptops, tiny-GPU air-gap, GGUF quantized  | `llama-cpp-python` in-process  |
| `MockBackend`      | Tests / CI — never touches a network           | None                           |

All SDK imports are **lazy**: missing optional deps don't break
`import dgx_ts_lab.llm`. SDKs are only loaded on first `generate()`.

## Files

| File                       | Purpose                                                                  |
|----------------------------|--------------------------------------------------------------------------|
| `backend.py`               | `LLMBackend` Protocol, Message/ToolDef/ToolCall/GenerateOptions/GenerateResult |
| `_mock_backend.py`         | Deterministic test double                                                 |
| `anthropic_backend.py`     | Claude (prompt caching + tool use)                                         |
| `vllm_backend.py`          | OpenAI-compatible HTTP client                                              |
| `ollama_backend.py`        | Native Ollama HTTP client                                                  |
| `llama_cpp_backend.py`     | In-process llama-cpp-python                                                |
| `factory.py`               | `build_backend(kind, **kwargs)` dispatcher                                 |
| `rag.py`                   | Numpy-only cosine retriever (TF-IDF lexical + optional embedded)            |
| `telemetry_tools.py`       | 4 default tools: query_telemetry, query_anomaly_history, lookup_procedure, read_model_card |
| `copilot.py`               | Multi-turn chat orchestrator with tool-call loop                            |
| `report_generator.py`      | B7 — LLM polish over Phase 7 explanation skeleton                          |
| `procedure_synth.py`       | B8 — NL → command sequence with validator retry loop                       |

## Hot-swap example

```python
from dgx_ts_lab.llm import build_backend
from dgx_ts_lab.llm.copilot import Copilot
from dgx_ts_lab.llm.telemetry_tools import default_tool_registry, CopilotContext

backend = build_backend("anthropic")            # or "vllm" / "ollama" / "llama_cpp" / "mock"
ctx = CopilotContext(telemetry=tel_array, channel_names=["bus_v", "bus_i"])
copilot = Copilot(backend=backend, tools=default_tool_registry(ctx))
turn = copilot.chat("how is bus_v over the last hour?")
print(turn.text)
```

The same script runs against any backend by changing one string.

## CLI

```bash
dgx-ts copilot --backend mock                    # smoke
dgx-ts copilot --backend anthropic
dgx-ts copilot --backend ollama --model llama3.1:8b
dgx-ts copilot --backend vllm --model meta-llama/Llama-3.1-70B-Instruct
dgx-ts copilot --backend llama_cpp --model data/llm_weights/mistral-7b.gguf
dgx-ts copilot --backend anthropic --procedures docs/procedures/ --model-card runs/last/model_card.yaml
```

## Tool-use loop (`Copilot._tool_loop`)

```
user_text
   │
   ▼
backend.generate(history, tools)
   │
   ├── no tool_calls → return text  ◀── exit
   │
   └── tool_calls present:
         append assistant msg (with tool_calls) to history
         for each call:
            tool_result = registry.invoke(call)
            append ToolResultMessage to history
         loop (capped by max_tool_iters)
```

Capped at `max_tool_iters=6` by default to bound runaway loops.

## Procedure synthesis loop (`ProcedureSynthesizer.synthesize`)

```
request
   │
   ▼
LLM returns JSON {"steps": [...]} (system prompt enforces shape)
   │
   ├── parse error → re-prompt with parse error
   ├── unknown opcode → re-prompt with vocab error
   ├── validator(steps) returns errors → re-prompt with sim errors
   └── all clean → return ProcedureSynthResult(success=True, ...)
```

Validator is user-supplied; Phase 8 `command_sequence_gen` makes a
natural plug-in.

## Tests

`packages/dgx_ts_lab/tests/test_phase11_llm.py` covers all of the above
with `MockBackend`. The optional `test_anthropic_live_smoke` test is
gated behind `ANTHROPIC_API_KEY` and is skipped in CI.
