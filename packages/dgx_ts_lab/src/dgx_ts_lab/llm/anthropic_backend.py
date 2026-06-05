"""AnthropicBackend — Claude via the official anthropic SDK.

Supports prompt caching (system + last few turns) and the Anthropic tool-use
format. Online only; for air-gap use VLLMBackend / OllamaBackend /
LlamaCppBackend.

Lazy import: the anthropic SDK is only imported on first generate(), so
`import dgx_ts_lab.llm` works without the package installed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from .backend import (
    GenerateOptions,
    GenerateResult,
    Message,
    Role,
    ToolCall,
    ToolDef,
)


_DEFAULT_MODEL = "claude-sonnet-4-5"


def _to_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Returns (system_text, message_list) in Anthropic API shape."""
    system_text = ""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == Role.SYSTEM:
            system_text += (("\n\n" if system_text else "") + m.content)
            continue
        if m.role == Role.USER:
            out.append({"role": "user", "content": m.content})
        elif m.role == Role.ASSISTANT:
            content_blocks: list[dict[str, Any]] = []
            if m.content:
                content_blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            out.append({"role": "assistant", "content": content_blocks or m.content})
        elif m.role == Role.TOOL:
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content,
                        }
                    ],
                }
            )
    return system_text, out


def _to_anthropic_tools(tools: list[ToolDef] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


class AnthropicBackend:
    name: str = "anthropic"

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "AnthropicBackend requires the 'anthropic' package: "
                "`pip install anthropic`"
            ) from e
        if not self._api_key:
            raise RuntimeError(
                "AnthropicBackend needs ANTHROPIC_API_KEY (env or constructor)"
            )
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        options: GenerateOptions | None = None,
    ) -> GenerateResult:
        client = self._ensure_client()
        options = options or GenerateOptions()
        system_text, msg_list = _to_anthropic_messages(messages)
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            messages=msg_list,
        )
        if system_text:
            if options.enable_prompt_cache:
                # Apply ephemeral cache_control to system block
                kwargs["system"] = [
                    {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)
        if options.stop:
            kwargs["stop_sequences"] = options.stop

        resp = client.messages.create(**kwargs)

        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_chunks.append(block.text)
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        usage = {}
        if hasattr(resp, "usage"):
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", 0),
                "output_tokens": getattr(resp.usage, "output_tokens", 0),
            }

        return GenerateResult(
            text="".join(text_chunks),
            tool_calls=tool_calls,
            finish_reason=getattr(resp, "stop_reason", "stop") or "stop",
            usage=usage,
            raw=resp,
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        options: GenerateOptions | None = None,
    ) -> Iterator[str]:
        client = self._ensure_client()
        options = options or GenerateOptions()
        system_text, msg_list = _to_anthropic_messages(messages)
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            messages=msg_list,
        )
        if system_text:
            kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
