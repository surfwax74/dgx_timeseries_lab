"""OllamaBackend — talks to a local Ollama daemon over its native HTTP API.

Easiest setup: `ollama serve` + `ollama pull llama3:8b`. Best fit for
RTX 3080 / single-A5000 dev machines. Air-gap friendly once the model is
pulled.

API ref: https://github.com/ollama/ollama/blob/main/docs/api.md
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


def _to_ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
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
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in m.tool_calls
                ]
            out.append(entry)
        elif m.role == Role.TOOL:
            out.append({"role": "tool", "content": m.content})
    return out


def _to_ollama_tools(tools: list[ToolDef] | None) -> list[dict[str, Any]]:
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


class OllamaBackend:
    name: str = "ollama"

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError(
                "OllamaBackend requires 'httpx': `pip install httpx`"
            ) from e
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout_s)
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
            "messages": _to_ollama_messages(messages),
            "stream": False,
            "options": {
                "temperature": options.temperature,
                "top_p": options.top_p,
                "num_predict": options.max_tokens,
            },
        }
        if tools:
            body["tools"] = _to_ollama_tools(tools)
        if options.stop:
            body["options"]["stop"] = options.stop

        resp = client.post("/api/chat", json=body)
        resp.raise_for_status()
        payload = resp.json()
        msg = payload.get("message", {})
        text = msg.get("content") or ""
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(
                ToolCall(id=str(uuid.uuid4()), name=fn.get("name", ""), arguments=args)
            )
        return GenerateResult(
            text=text,
            tool_calls=tool_calls,
            finish_reason="stop" if payload.get("done") else "length",
            usage={
                "input_tokens": payload.get("prompt_eval_count", 0),
                "output_tokens": payload.get("eval_count", 0),
            },
            raw=payload,
        )

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
            "messages": _to_ollama_messages(messages),
            "stream": True,
            "options": {
                "temperature": options.temperature,
                "top_p": options.top_p,
                "num_predict": options.max_tokens,
            },
        }
        if tools:
            body["tools"] = _to_ollama_tools(tools)
        with client.stream("POST", "/api/chat", json=body) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "message" in chunk and chunk["message"].get("content"):
                    yield chunk["message"]["content"]
                if chunk.get("done"):
                    break
