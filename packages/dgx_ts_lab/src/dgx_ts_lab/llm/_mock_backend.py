"""MockBackend — deterministic test double conforming to LLMBackend.

Lets the co-pilot, report generator, and procedure synthesizer be tested
without an LLM SDK / server / API key. Two modes:

    * scripted   — return a queued list of GenerateResult objects in order
    * echo       — return the last user message back, capitalized

Also records the request log so tests can assert on prompts.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .backend import (
    GenerateOptions,
    GenerateResult,
    Message,
    Role,
    ToolDef,
)


class MockBackend:
    name: str = "mock"
    model_id: str = "mock-1"

    def __init__(
        self,
        scripted_results: list[GenerateResult] | None = None,
        echo: bool = False,
    ) -> None:
        self._queue = list(scripted_results or [])
        self._echo = bool(echo)
        self.call_log: list[dict[str, Any]] = []

    # ── LLMBackend ──────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        options: GenerateOptions | None = None,
    ) -> GenerateResult:
        self.call_log.append(
            {
                "messages": [
                    {"role": m.role.value, "content": m.content} for m in messages
                ],
                "tools": [t.name for t in (tools or [])],
                "options": options,
            }
        )
        if self._queue:
            return self._queue.pop(0)
        if self._echo:
            last_user = next(
                (m for m in reversed(messages) if m.role == Role.USER),
                None,
            )
            text = (last_user.content if last_user else "").upper() or "MOCK"
            return GenerateResult(text=text, finish_reason="stop")
        return GenerateResult(text="[mock]", finish_reason="stop")

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        options: GenerateOptions | None = None,
    ) -> Iterator[str]:
        # Stream the same content as generate(), one word at a time
        result = self.generate(messages, tools, options)
        for word in result.text.split():
            yield word + " "
