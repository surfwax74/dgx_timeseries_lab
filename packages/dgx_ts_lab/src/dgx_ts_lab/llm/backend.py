"""LLMBackend Protocol — the central abstraction for Phase 11.

Four concrete backends (Anthropic, vLLM, Ollama, llama.cpp) all implement
the same interface so the co-pilot, report generator, and procedure
synthesizer don't care which one is plugged in.

Message types are deliberately structural — a small dataclass family
covering system / user / assistant / tool-result. Tool definitions are
JSONSchema-shaped so they pass through every provider unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class Role(StrEnum):
    """Standard chat roles. Stored as str so JSON serialization is trivial."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A tool the assistant wants invoked. ID is provider-assigned."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """Generic message. Use the role-specific helpers below for clarity."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None       # set on TOOL messages
    name: str | None = None                # tool name on TOOL messages


def SystemMessage(content: str) -> Message:
    return Message(role=Role.SYSTEM, content=content)


def UserMessage(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def AssistantMessage(content: str, tool_calls: list[ToolCall] | None = None) -> Message:
    return Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or [])


def ToolResultMessage(tool_call_id: str, name: str, content: str) -> Message:
    return Message(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class ToolDef:
    """JSONSchema-shaped tool definition. Same shape across all backends."""

    name: str
    description: str
    parameters: dict[str, Any]   # JSONSchema for arguments


@dataclass
class GenerateOptions:
    """Per-call generation parameters. Defaults match production hygiene."""

    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)
    enable_prompt_cache: bool = True       # honored by AnthropicBackend


@dataclass
class GenerateResult:
    """One completion. `tool_calls` is non-empty when the model wants tools."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"            # stop | length | tool_use | error
    usage: dict[str, int] = field(default_factory=dict)  # input_tokens, output_tokens
    raw: Any = None                         # provider-specific response object


@runtime_checkable
class LLMBackend(Protocol):
    """The single interface co-pilot code talks to.

    All four concrete backends (Anthropic / vLLM / Ollama / llama.cpp) plus
    the MockBackend conform. ``name`` is for run logging; ``model_id`` is
    the provider-specific identifier (Claude model name, GGUF filename,
    Ollama tag, etc.).
    """

    name: str
    model_id: str

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        options: GenerateOptions | None = None,
    ) -> GenerateResult:
        """Produce one completion. Synchronous; co-pilot orchestrator
        handles tool-call loops on top of repeated calls."""
        ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        options: GenerateOptions | None = None,
    ) -> Iterator[str]:
        """Yield content chunks. Tool-using turns may yield empty strings
        and rely on the caller to re-call ``generate`` to materialize
        tool_calls — kept simple for air-gap UX."""
        ...
