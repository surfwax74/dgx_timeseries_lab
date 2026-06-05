"""VLLMBackend — talks to a vLLM HTTP server (OpenAI-compatible API).

Best for H200 / A5000xN tiers where vLLM's PagedAttention + continuous
batching pay off. Air-gap friendly: the vLLM server can run entirely on
the DGX with no external network.

Boot the server with `scripts/setup_vllm_server.sh`.
"""

from __future__ import annotations

import json
import uuid
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


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert our Message list to OpenAI ChatCompletions shape."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == Role.SYSTEM:
            out.append({"role": "system", "content": m.content})
        elif m.role == Role.USER:
            out.append({"role": "user", "content": m.content})
        elif m.role == Role.ASSISTANT:
            entry: dict[str, Any] = {"role": "assistant", "content": m.content}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(entry)
        elif m.role == Role.TOOL:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "name": m.name,
                    "content": m.content,
                }
            )
    return out


def _to_openai_tools(tools: list[ToolDef] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _parse_openai_response(payload: dict[str, Any]) -> GenerateResult:
    choices = payload.get("choices") or [{}]
    msg = choices[0].get("message", {})
    text = msg.get("content") or ""
    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        tool_calls.append(
            ToolCall(id=tc.get("id") or str(uuid.uuid4()), name=fn.get("name", ""), arguments=args)
        )
    usage = payload.get("usage") or {}
    return GenerateResult(
        text=text,
        tool_calls=tool_calls,
        finish_reason=choices[0].get("finish_reason", "stop"),
        usage={
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
        raw=payload,
    )


class VLLMBackend:
    """OpenAI-compatible HTTP client for a local vLLM server."""

    name: str = "vllm"

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        timeout_s: float = 120.0,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError(
                "VLLMBackend requires 'httpx': `pip install httpx`"
            ) from e
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_s,
        )
        return self._client

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        options: GenerateOptions | None = None,
    ) -> GenerateResult:
        client = self._ensure_client()
        options = options or GenerateOptions()
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": _to_openai_messages(messages),
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
            "top_p": options.top_p,
        }
        if tools:
            body["tools"] = _to_openai_tools(tools)
            body["tool_choice"] = "auto"
        if options.stop:
            body["stop"] = options.stop
        resp = client.post("/chat/completions", json=body)
        resp.raise_for_status()
        return _parse_openai_response(resp.json())

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        options: GenerateOptions | None = None,
    ) -> Iterator[str]:
        client = self._ensure_client()
        options = options or GenerateOptions()
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": _to_openai_messages(messages),
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "stream": True,
        }
        if tools:
            body["tools"] = _to_openai_tools(tools)
            body["tool_choice"] = "auto"
        with client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        yield delta["content"]
