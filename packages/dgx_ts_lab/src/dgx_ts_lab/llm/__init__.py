"""Phase 11 — LLM ops co-pilot.

Quad-backend LLM abstraction so the same co-pilot code runs against:

    * AnthropicBackend   — Claude (online; needs ANTHROPIC_API_KEY)
    * VLLMBackend        — vLLM HTTP server (production air-gap on H200)
    * OllamaBackend      — Ollama HTTP daemon (single-GPU dev / RTX 3080)
    * LlamaCppBackend    — llama-cpp-python in-process (CPU / tiny GPU)
    * MockBackend        — deterministic test double (no SDK / no server)

All backends conform to the `LLMBackend` Protocol in `backend.py`. SDKs are
imported lazily — missing optional deps don't break `import dgx_ts_lab`.

The co-pilot layer (`copilot.py`, `report_generator.py`,
`procedure_synth.py`) only talks to `LLMBackend`. Switching backends is a
one-line config swap; nothing downstream changes.
"""

from __future__ import annotations

from .backend import (
    AssistantMessage,
    GenerateOptions,
    GenerateResult,
    LLMBackend,
    Message,
    Role,
    SystemMessage,
    ToolCall,
    ToolDef,
    ToolResultMessage,
    UserMessage,
)
from ._mock_backend import MockBackend
from .factory import build_backend

__all__ = [
    "AssistantMessage",
    "GenerateOptions",
    "GenerateResult",
    "LLMBackend",
    "Message",
    "MockBackend",
    "Role",
    "SystemMessage",
    "ToolCall",
    "ToolDef",
    "ToolResultMessage",
    "UserMessage",
    "build_backend",
]
