"""Phase 11 tests — LLM ops co-pilot.

Coverage:
    * Message + ToolDef + GenerateResult shapes
    * MockBackend conforms to LLMBackend Protocol
    * Anthropic / vLLM / Ollama / llama.cpp factory dispatch (no SDK calls)
    * Anthropic + OpenAI message-shape translators (unit-level)
    * RAG: TF-IDF lexical index returns top-k by cosine
    * Tool registry: invoke routes to fn, errors are stringified
    * Copilot tool-use loop drives queued MockBackend results
    * ReportGenerator passes the skeleton + asks for a polish
    * ProcedureSynthesizer parses + validates + retries on errors

No live SDK / server / API key is required; everything runs against
MockBackend.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from dgx_ts_lab.llm import (
    AssistantMessage,
    GenerateOptions,
    GenerateResult,
    LLMBackend,
    Message,
    MockBackend,
    Role,
    SystemMessage,
    ToolCall,
    ToolDef,
    ToolResultMessage,
    UserMessage,
    build_backend,
)


# ── Message / Result types ─────────────────────────────────────────────


def test_message_roles_and_helpers() -> None:
    s = SystemMessage("sys")
    u = UserMessage("hi")
    a = AssistantMessage("hello", tool_calls=[ToolCall("id1", "query", {})])
    t = ToolResultMessage("id1", "query", "{}")
    assert s.role == Role.SYSTEM
    assert u.role == Role.USER
    assert a.role == Role.ASSISTANT and len(a.tool_calls) == 1
    assert t.role == Role.TOOL and t.tool_call_id == "id1"


def test_generate_result_defaults() -> None:
    r = GenerateResult(text="ok")
    assert r.text == "ok"
    assert r.tool_calls == []
    assert r.finish_reason == "stop"


# ── MockBackend conforms to Protocol ────────────────────────────────────


def test_mock_backend_conforms_to_protocol() -> None:
    mb = MockBackend()
    assert isinstance(mb, LLMBackend)


def test_mock_backend_echo_mode() -> None:
    mb = MockBackend(echo=True)
    res = mb.generate([UserMessage("hello world")])
    assert res.text == "HELLO WORLD"


def test_mock_backend_scripted() -> None:
    queue = [GenerateResult(text="one"), GenerateResult(text="two")]
    mb = MockBackend(scripted_results=queue)
    assert mb.generate([UserMessage("a")]).text == "one"
    assert mb.generate([UserMessage("b")]).text == "two"
    # Falls through to default after queue drained
    assert mb.generate([UserMessage("c")]).text == "[mock]"


def test_mock_backend_records_call_log() -> None:
    mb = MockBackend()
    mb.generate([SystemMessage("s"), UserMessage("u")])
    assert len(mb.call_log) == 1
    assert mb.call_log[0]["messages"][0]["role"] == "system"


# ── Factory dispatch (no SDK touched) ──────────────────────────────────


def test_factory_mock() -> None:
    b = build_backend("mock")
    assert isinstance(b, MockBackend)


def test_factory_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown LLM backend"):
        build_backend("nope")


def test_factory_anthropic_constructs_without_sdk_use() -> None:
    # Construction is allowed; the SDK is only imported on first generate().
    from dgx_ts_lab.llm.anthropic_backend import AnthropicBackend
    b = AnthropicBackend(api_key="dummy")
    assert b.name == "anthropic"
    assert b.model_id.startswith("claude")


def test_factory_vllm_constructs_without_sdk_use() -> None:
    from dgx_ts_lab.llm.vllm_backend import VLLMBackend
    b = VLLMBackend(model_id="meta-llama/Llama-3.1-8B-Instruct")
    assert b.name == "vllm" and b.base_url.endswith("/v1")


def test_factory_ollama_constructs_without_sdk_use() -> None:
    from dgx_ts_lab.llm.ollama_backend import OllamaBackend
    b = OllamaBackend(model_id="llama3.1:8b")
    assert b.name == "ollama" and b.base_url.endswith(":11434")


def test_factory_llama_cpp_constructs_without_loading_model() -> None:
    from dgx_ts_lab.llm.llama_cpp_backend import LlamaCppBackend
    b = LlamaCppBackend(model_path="/nonexistent.gguf")
    assert b.name == "llama_cpp" and b.model_id == "nonexistent.gguf"


# ── Message translators (unit-level) ────────────────────────────────────


def test_anthropic_message_translator_handles_all_roles() -> None:
    from dgx_ts_lab.llm.anthropic_backend import _to_anthropic_messages

    msgs = [
        SystemMessage("sys"),
        UserMessage("u1"),
        AssistantMessage("a1", tool_calls=[ToolCall("c1", "query", {"x": 1})]),
        ToolResultMessage("c1", "query", "{}"),
    ]
    system_text, out = _to_anthropic_messages(msgs)
    assert system_text == "sys"
    assert out[0]["role"] == "user"
    assert out[1]["role"] == "assistant"
    # The tool_use block should be present in the assistant turn
    blocks = out[1]["content"]
    assert any(b.get("type") == "tool_use" for b in blocks)
    # Tool result becomes a user turn with tool_result block
    assert out[2]["role"] == "user"
    assert out[2]["content"][0]["type"] == "tool_result"


def test_vllm_message_translator_handles_tool_call_args_as_json() -> None:
    from dgx_ts_lab.llm.vllm_backend import _to_openai_messages

    msgs = [AssistantMessage("", tool_calls=[ToolCall("c1", "query", {"x": 1})])]
    out = _to_openai_messages(msgs)
    assert out[0]["tool_calls"][0]["function"]["name"] == "query"
    # OpenAI shape requires arguments to be a JSON string
    assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"x": 1}


# ── RAG ─────────────────────────────────────────────────────────────────


def test_rag_lexical_returns_relevant_doc_first() -> None:
    from dgx_ts_lab.llm.rag import CosineRAGIndex, RAGDocument

    idx = CosineRAGIndex()
    docs = [
        RAGDocument("p1", "Eclipse safe mode",
                    "Enter safe mode when bus voltage drops during eclipse."),
        RAGDocument("p2", "Payload activation",
                    "Activate payload only when battery state of charge exceeds 80%."),
        RAGDocument("p3", "Comms reset",
                    "Power-cycle the transponder when downlink errors exceed threshold."),
    ]
    idx.add_lexical(docs)
    hits = idx.query("how do I enter safe mode during eclipse", top_k=2)
    assert hits[0].document.doc_id == "p1"
    assert hits[0].score > 0.0


def test_rag_load_procedures_directory(tmp_path: Path) -> None:
    from dgx_ts_lab.llm.rag import load_procedures_directory

    (tmp_path / "a.md").write_text("# Eclipse\nDo X.", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# Battery\nDo Y.", encoding="utf-8")
    docs = load_procedures_directory(tmp_path)
    titles = [d.title for d in docs]
    assert "a" in titles and "b" in titles


def test_rag_save_load_roundtrip(tmp_path: Path) -> None:
    from dgx_ts_lab.llm.rag import CosineRAGIndex, RAGDocument

    idx = CosineRAGIndex()
    docs = [RAGDocument(f"p{i}", f"t{i}", f"text {i} eclipse mode") for i in range(3)]
    idx.add_lexical(docs)
    p = tmp_path / "idx.npz"
    idx.save(p)
    loaded = CosineRAGIndex.load(p)
    assert loaded.n_docs == 3
    # NB: lexical re-query needs the vectorizer; loaded one only stores vectors.
    # Confirm structural fields survived.
    assert loaded._docs[0].doc_id == "p0"


# ── Telemetry tools ─────────────────────────────────────────────────────


def _make_context_with_telemetry():
    from dgx_ts_lab.llm.rag import CosineRAGIndex, RAGDocument
    from dgx_ts_lab.llm.telemetry_tools import CopilotContext

    tel = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(100, 1)
    scores = np.zeros(100, dtype=np.float32)
    scores[50] = 5.0
    scores[75] = 3.0
    rag = CosineRAGIndex()
    rag.add_lexical(
        [
            RAGDocument("p1", "Safe mode", "Enter safe mode during anomalies."),
            RAGDocument("p2", "Routine", "Routine ops checklist."),
        ]
    )
    return CopilotContext(
        telemetry=tel,
        channel_names=["bus_v"],
        anomaly_scores=scores,
        anomaly_threshold=1.0,
        rag_index=rag,
    )


def test_tool_registry_query_telemetry() -> None:
    from dgx_ts_lab.llm.telemetry_tools import default_tool_registry

    reg = default_tool_registry(context=_make_context_with_telemetry())
    out = reg.invoke(
        ToolCall(id="x", name="query_telemetry", arguments={"channel": "bus_v"})
    )
    parsed = json.loads(out)
    assert parsed["channel"] == "bus_v"
    assert parsed["n_samples"] == 100


def test_tool_registry_query_anomaly_history() -> None:
    from dgx_ts_lab.llm.telemetry_tools import default_tool_registry

    reg = default_tool_registry(context=_make_context_with_telemetry())
    out = reg.invoke(
        ToolCall(id="x", name="query_anomaly_history", arguments={"top_k": 5})
    )
    parsed = json.loads(out)
    assert parsed["n_above_threshold"] == 2
    # Highest-scoring event should come first
    assert parsed["events"][0]["step"] == 50


def test_tool_registry_lookup_procedure() -> None:
    from dgx_ts_lab.llm.telemetry_tools import default_tool_registry

    reg = default_tool_registry(context=_make_context_with_telemetry())
    out = reg.invoke(
        ToolCall(id="x", name="lookup_procedure",
                 arguments={"query": "anomaly safe mode", "top_k": 1})
    )
    parsed = json.loads(out)
    assert parsed["hits"][0]["doc_id"] == "p1"


def test_tool_registry_unknown_tool_returns_error() -> None:
    from dgx_ts_lab.llm.telemetry_tools import default_tool_registry

    reg = default_tool_registry(context=_make_context_with_telemetry())
    out = reg.invoke(ToolCall(id="x", name="nope", arguments={}))
    assert out.startswith("ERROR: unknown tool")


def test_tool_registry_read_model_card(tmp_path: Path) -> None:
    from dgx_ts_lab.llm.telemetry_tools import CopilotContext, default_tool_registry

    card = tmp_path / "card.yaml"
    card.write_text("name: dummy\nversion: 1\n", encoding="utf-8")
    reg = default_tool_registry(context=CopilotContext(model_card_path=card))
    out = reg.invoke(ToolCall(id="x", name="read_model_card", arguments={}))
    assert "name: dummy" in out


# ── Copilot orchestrator ────────────────────────────────────────────────


def test_copilot_runs_simple_turn_without_tools() -> None:
    from dgx_ts_lab.llm.copilot import Copilot
    from dgx_ts_lab.llm.telemetry_tools import default_tool_registry

    backend = MockBackend(scripted_results=[GenerateResult(text="hello back")])
    cop = Copilot(backend=backend, tools=default_tool_registry())
    turn = cop.chat("hi")
    assert turn.text == "hello back"
    assert turn.n_tool_iterations == 0


def test_copilot_runs_tool_loop_then_summarizes() -> None:
    from dgx_ts_lab.llm.copilot import Copilot
    from dgx_ts_lab.llm.telemetry_tools import default_tool_registry

    # First turn asks the LLM, LLM returns a tool call.
    # Second turn (after we feed the tool result back), LLM returns the final text.
    queued = [
        GenerateResult(
            text="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="query_telemetry",
                    arguments={"channel": "bus_v"},
                )
            ],
            finish_reason="tool_use",
        ),
        GenerateResult(text="bus_v stats look nominal", finish_reason="stop"),
    ]
    backend = MockBackend(scripted_results=queued)
    cop = Copilot(
        backend=backend,
        tools=default_tool_registry(context=_make_context_with_telemetry()),
    )
    turn = cop.chat("how is bus_v?")
    assert turn.text == "bus_v stats look nominal"
    assert turn.n_tool_iterations == 1
    assert turn.tool_calls_made[0]["name"] == "query_telemetry"


def test_copilot_reset_clears_history() -> None:
    from dgx_ts_lab.llm.copilot import Copilot
    from dgx_ts_lab.llm.telemetry_tools import default_tool_registry

    cop = Copilot(backend=MockBackend(), tools=default_tool_registry())
    cop.chat("first")
    assert len(cop.history) > 1
    cop.reset()
    assert len(cop.history) == 1  # just the system prompt


# ── Report generator ────────────────────────────────────────────────────


def test_report_generator_returns_polished_markdown() -> None:
    from dgx_ts_lab.llm.report_generator import ReportGenerator
    from dgx_ts_lab.llm.telemetry_tools import default_tool_registry

    backend = MockBackend(
        scripted_results=[
            GenerateResult(
                text="# Executive Summary\nLooks like an EPS bus spike.\n"
                     "\n(skeleton retained)\n\n# Recommended Actions\n- Check fuse.",
                finish_reason="stop",
            )
        ]
    )
    rg = ReportGenerator(backend=backend, tools=default_tool_registry())
    out = rg.polish("# Raw Anomaly Report\n- channel: bus_v\n- step: 1234\n")
    assert "Executive Summary" in out.markdown
    assert "Recommended Actions" in out.markdown


# ── Procedure synthesizer ──────────────────────────────────────────────


def test_procedure_synth_happy_path() -> None:
    from dgx_ts_lab.llm.procedure_synth import ProcedureSynthesizer

    backend = MockBackend(
        scripted_results=[
            GenerateResult(
                text='{"steps": [{"opcode": "TELEM_REQUEST", '
                     '"params": {"target": "EPS"}, '
                     '"rationale": "snapshot before action"}]}'
            )
        ]
    )
    synth = ProcedureSynthesizer(
        backend=backend,
        opcodes=["TELEM_REQUEST", "MODE_SET"],
        param_values=["EPS", "NORMAL"],
        validator=lambda _: [],
    )
    res = synth.synthesize("grab a telemetry snapshot")
    assert res.success
    assert len(res.steps) == 1
    assert res.steps[0].opcode == "TELEM_REQUEST"


def test_procedure_synth_retries_on_validator_errors() -> None:
    from dgx_ts_lab.llm.procedure_synth import ProcedureSynthesizer

    # First attempt uses a forbidden opcode, second corrects it
    backend = MockBackend(
        scripted_results=[
            GenerateResult(text='{"steps": [{"opcode": "EVIL_OP", "params": {}}]}'),
            GenerateResult(
                text='{"steps": [{"opcode": "MODE_SET", "params": {"to": "SAFE"}}]}'
            ),
        ]
    )
    synth = ProcedureSynthesizer(
        backend=backend,
        opcodes=["TELEM_REQUEST", "MODE_SET"],
        param_values=["SAFE"],
        validator=lambda _: [],
        max_validation_iters=2,
    )
    res = synth.synthesize("switch to safe mode")
    assert res.success
    assert res.n_iterations == 2
    assert res.steps[0].opcode == "MODE_SET"


def test_procedure_synth_fails_after_max_iterations() -> None:
    from dgx_ts_lab.llm.procedure_synth import ProcedureSynthesizer

    backend = MockBackend(
        scripted_results=[
            GenerateResult(text='{"steps": [{"opcode": "TELEM_REQUEST"}]}'),
            GenerateResult(text='{"steps": [{"opcode": "TELEM_REQUEST"}]}'),
        ]
    )
    synth = ProcedureSynthesizer(
        backend=backend,
        opcodes=["TELEM_REQUEST"],
        param_values=[],
        validator=lambda _: ["always invalid"],
        max_validation_iters=1,
    )
    res = synth.synthesize("anything")
    assert not res.success
    assert "always invalid" in res.validation_errors[0]


def test_procedure_synth_parse_error_recovers() -> None:
    from dgx_ts_lab.llm.procedure_synth import ProcedureSynthesizer

    backend = MockBackend(
        scripted_results=[
            GenerateResult(text="this is not JSON at all"),
            GenerateResult(
                text='{"steps": [{"opcode": "TELEM_REQUEST", "params": {}}]}'
            ),
        ]
    )
    synth = ProcedureSynthesizer(
        backend=backend,
        opcodes=["TELEM_REQUEST"],
        param_values=[],
        validator=lambda _: [],
        max_validation_iters=2,
    )
    res = synth.synthesize("snapshot")
    assert res.success
    assert res.n_iterations == 2


def test_procedure_synth_handles_fenced_json() -> None:
    from dgx_ts_lab.llm.procedure_synth import ProcedureSynthesizer

    backend = MockBackend(
        scripted_results=[
            GenerateResult(
                text='```json\n{"steps": [{"opcode": "TELEM_REQUEST"}]}\n```'
            )
        ]
    )
    synth = ProcedureSynthesizer(
        backend=backend,
        opcodes=["TELEM_REQUEST"],
        param_values=[],
        validator=lambda _: [],
    )
    res = synth.synthesize("snapshot")
    assert res.success


# ── Optional live-API integration (skipped without env key) ────────────


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_anthropic_live_smoke() -> None:
    """Tiny live call; skipped in CI unless a key is supplied."""
    backend = build_backend("anthropic")
    res = backend.generate(
        [SystemMessage("Reply 'pong'."), UserMessage("ping")],
        options=GenerateOptions(max_tokens=8),
    )
    assert res.text  # non-empty
