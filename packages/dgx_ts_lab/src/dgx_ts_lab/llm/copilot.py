"""Co-pilot orchestrator — multi-turn chat with tool-use loop.

Backend-agnostic: takes any `LLMBackend` + a `ToolRegistry` + an
optional system prompt. Maintains chat history; on each user turn:

    1. send history + tools to the LLM
    2. if the response has tool_calls, run them via the registry
    3. append the tool results, re-call the LLM
    4. repeat until no more tool_calls (capped by max_tool_iters)
    5. return final assistant text + updated history

Designed for the `dgx-ts copilot` REPL but also importable for batch use
(e.g., ReportGenerator delegates to it for the "polish the report" turn).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .backend import (
    AssistantMessage,
    GenerateOptions,
    GenerateResult,
    LLMBackend,
    Message,
    Role,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from .telemetry_tools import ToolRegistry


_DEFAULT_SYSTEM = (
    "You are a satellite operations co-pilot. You answer questions about "
    "telemetry, anomalies, and mission procedures. When you need data, "
    "call the available tools. Be concise; cite tool results explicitly."
)


@dataclass
class CopilotTurnResult:
    """Result of one user turn — final assistant text + all tool calls
    that fired this turn (for logging / replay)."""

    text: str
    n_tool_iterations: int = 0
    tool_calls_made: list[dict] = field(default_factory=list)
    raw_final: GenerateResult | None = None


class Copilot:
    """Stateful multi-turn chat with tool-use loop."""

    def __init__(
        self,
        backend: LLMBackend,
        tools: ToolRegistry,
        system_prompt: str | None = None,
        max_tool_iters: int = 6,
        options: GenerateOptions | None = None,
    ) -> None:
        self.backend = backend
        self.tools = tools
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM
        self.max_tool_iters = int(max_tool_iters)
        self.options = options or GenerateOptions()
        self.history: list[Message] = [SystemMessage(self.system_prompt)]

    # ── Single-turn API ─────────────────────────────────────────────────

    def chat(self, user_text: str) -> CopilotTurnResult:
        """Run one user→assistant turn, including any tool calls."""
        self.history.append(UserMessage(user_text))
        result = self._tool_loop()
        self.history.append(AssistantMessage(result.text))
        return result

    def _tool_loop(self) -> CopilotTurnResult:
        n_iter = 0
        tool_log: list[dict] = []
        tool_defs = self.tools.list_defs()
        last_text = ""
        last_raw: Optional[GenerateResult] = None
        for _ in range(self.max_tool_iters + 1):
            result = self.backend.generate(
                self.history, tools=tool_defs, options=self.options,
            )
            last_raw = result
            last_text = result.text
            if not result.tool_calls:
                break
            # Record assistant turn including tool_calls so the LLM sees its own request
            self.history.append(
                AssistantMessage(result.text, tool_calls=list(result.tool_calls))
            )
            for tc in result.tool_calls:
                out = self.tools.invoke(tc)
                tool_log.append(
                    {"name": tc.name, "args": tc.arguments, "result_chars": len(out)}
                )
                self.history.append(
                    ToolResultMessage(tool_call_id=tc.id, name=tc.name, content=out)
                )
            n_iter += 1
        return CopilotTurnResult(
            text=last_text,
            n_tool_iterations=n_iter,
            tool_calls_made=tool_log,
            raw_final=last_raw,
        )

    # ── Convenience ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop chat history (keeps system prompt)."""
        self.history = [SystemMessage(self.system_prompt)]

    def export_history(self) -> list[dict]:
        """Plain-dict export for logging / MLflow artifacts."""
        return [
            {"role": m.role.value, "content": m.content}
            for m in self.history
            if m.role in (Role.SYSTEM, Role.USER, Role.ASSISTANT)
        ]
