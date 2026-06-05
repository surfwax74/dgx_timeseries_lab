"""LlamaCppBackend — in-process llama-cpp-python with GGUF quantized models.

Best for CPU-only laptops or RTX 3080 with quantized 7B/8B models. No
server process needed — runs in the same Python process.

Tool calling support depends on the underlying model template. We
implement it via llama-cpp-python's chat-handler-with-tools when
available, and gracefully degrade to text-only completion otherwise.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .backend import (
    GenerateOptions,
    GenerateResult,
    Message,
    Role,
    ToolCall,
    ToolDef,
)


def _to_llama_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """llama-cpp-python uses OpenAI-ish dict shape."""
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


class LlamaCppBackend:
    """Single-process llama-cpp-python backend.

    Required: pip install llama-cpp-python (CPU) or
    llama-cpp-python[cuda] (GPU). Pass `model_path` to a local .gguf file.
    """

    name: str = "llama_cpp"

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.model_id = Path(self.model_path).name
        self._n_ctx = int(n_ctx)
        self._n_gpu_layers = int(n_gpu_layers)
        self._n_threads = n_threads
        self._verbose = bool(verbose)
        self._llm: Any = None

    def _ensure_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "LlamaCppBackend requires 'llama-cpp-python'. "
                "Install with `pip install llama-cpp-python` (CPU) or "
                "`pip install llama-cpp-python[cuda]` (GPU)."
            ) from e
        kwargs: dict[str, Any] = dict(
            model_path=self.model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=self._verbose,
        )
        if self._n_threads is not None:
            kwargs["n_threads"] = int(self._n_threads)
        self._llm = Llama(**kwargs)
        return self._llm

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        options: GenerateOptions | None = None,
    ) -> GenerateResult:
        llm = self._ensure_llm()
        options = options or GenerateOptions()
        kwargs: dict[str, Any] = dict(
            messages=_to_llama_messages(messages),
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
        )
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        if options.stop:
            kwargs["stop"] = options.stop

        resp = llm.create_chat_completion(**kwargs)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message", {})
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
                ToolCall(
                    id=tc.get("id") or str(uuid.uuid4()),
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )
        usage = resp.get("usage") or {}
        return GenerateResult(
            text=text,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            raw=resp,
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        options: GenerateOptions | None = None,
    ) -> Iterator[str]:
        llm = self._ensure_llm()
        options = options or GenerateOptions()
        kwargs: dict[str, Any] = dict(
            messages=_to_llama_messages(messages),
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            stream=True,
        )
        if options.stop:
            kwargs["stop"] = options.stop
        for chunk in llm.create_chat_completion(**kwargs):
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            if delta.get("content"):
                yield delta["content"]
