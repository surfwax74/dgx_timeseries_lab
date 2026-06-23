"""B7 — LLM-polished anomaly report.

Takes a Phase 7 ExplanationReport (or any markdown skeleton) + the
co-pilot's tool registry, and asks the LLM to:

    1. summarize the event in operator language
    2. cite tool results inline ("query_anomaly_history showed peak at t=...")
    3. list recommended next actions

Returns the final markdown. Does NOT mutate the skeleton — the LLM
prepends an "Executive Summary" and appends a "Recommended Actions"
section.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import GenerateOptions, LLMBackend
from .copilot import Copilot
from .telemetry_tools import ToolRegistry

_REPORT_SYSTEM = (
    "You are a satellite-operations report writer. You are given a raw "
    "Markdown anomaly report skeleton produced by an automated explainer. "
    "Your job is to (1) write an Executive Summary at the top, (2) use the "
    "available tools to verify claims and pull in concrete numbers, and "
    "(3) add a 'Recommended Actions' section at the bottom. Do not remove "
    "or reorder any existing sections; only add new content."
)


_REPORT_USER_TEMPLATE = """\
Here is the automated anomaly report skeleton. Polish it per the system instructions.

```
{skeleton}
```

After your additions, return the complete final report as a single markdown document.
"""


@dataclass
class ReportPolishResult:
    """Polished report + provenance."""

    markdown: str
    n_tool_iterations: int
    tool_calls_made: list[dict]


class ReportGenerator:
    """LLM-driven polish pass over a Phase 7 explanation skeleton."""

    def __init__(
        self,
        backend: LLMBackend,
        tools: ToolRegistry,
        max_tool_iters: int = 8,
        options: GenerateOptions | None = None,
    ) -> None:
        self.backend = backend
        self.tools = tools
        self.max_tool_iters = int(max_tool_iters)
        self.options = options or GenerateOptions(max_tokens=2048, temperature=0.2)

    def polish(self, skeleton_markdown: str) -> ReportPolishResult:
        """Run the polish turn and return the final markdown."""
        copilot = Copilot(
            backend=self.backend,
            tools=self.tools,
            system_prompt=_REPORT_SYSTEM,
            max_tool_iters=self.max_tool_iters,
            options=self.options,
        )
        turn = copilot.chat(_REPORT_USER_TEMPLATE.format(skeleton=skeleton_markdown))
        return ReportPolishResult(
            markdown=turn.text,
            n_tool_iterations=turn.n_tool_iterations,
            tool_calls_made=turn.tool_calls_made,
        )
