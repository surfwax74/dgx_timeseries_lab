"""Execute capability-escalation scenarios against one or more backends.

Runs each scenario through the normal `Copilot` tool loop — the same
code path production uses — so results reflect real agentic behavior
rather than a bespoke harness that flatters the model.

Nothing here is model-aware. The runner does not know which backend is
"supposed" to win, and the graders are deterministic given a response.
That matters: the output has to survive someone asking "did you tune
this to make the big model look good?"
"""

from __future__ import annotations

# Doc IDs are the ``ABC-123`` prefix of each procedure filename, e.g.
# "EPS-201_undervoltage_response.md" -> "EPS-201". Used by the
# hallucination grader to decide whether a citation is real.
import re as _re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..copilot import Copilot
from ..rag import CosineRAGIndex, RAGDocument, load_procedures_directory
from ..telemetry_tools import CopilotContext, ToolRegistry, default_tool_registry
from .graders import GraderResult, ResponseUnderTest, run_grader
from .scenario import Scenario

_DOC_ID_FROM_NAME = _re.compile(r"^([A-Z]{3}-\d{3})")


@dataclass
class ScenarioRun:
    """Result of one (scenario x backend) cell."""

    scenario_id: str
    scenario_title: str
    tier: int
    backend_name: str
    model_id: str
    score: float
    passed: bool
    grader_results: list[GraderResult] = field(default_factory=list)
    response_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    n_tool_iterations: int = 0
    elapsed_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def corpus_doc_ids(procedures_dir: str | Path) -> set[str]:
    """Extract the set of legitimate ``ABC-123`` IDs from procedure filenames."""
    root = Path(procedures_dir)
    ids: set[str] = set()
    if not root.exists():
        return ids
    for p in root.rglob("*.md"):
        m = _DOC_ID_FROM_NAME.match(p.name)
        if m:
            ids.add(m.group(1))
    return ids


def build_rag_index(procedures_dir: str | Path) -> CosineRAGIndex:
    """Build a lexical RAG index over a directory of procedure markdown."""
    docs: list[RAGDocument] = load_procedures_directory(procedures_dir)
    index = CosineRAGIndex()
    if docs:
        index.add_lexical(docs)
    return index


def run_scenario(
    scenario: Scenario,
    backend: Any,
    tools: ToolRegistry,
    doc_ids: set[str],
    system_prompt: str | None = None,
) -> ScenarioRun:
    """Run one scenario against one backend and grade the result.

    A backend exception is captured rather than raised — one flaky model
    should not abort a matrix that takes an hour to produce.
    """
    copilot = Copilot(
        backend=backend,
        tools=tools,
        system_prompt=system_prompt,
        max_tool_iters=scenario.max_tool_iters,
    )

    t0 = time.time()
    try:
        turn = copilot.chat(scenario.prompt)
        elapsed = time.time() - t0
    except Exception as e:                                   # noqa: BLE001
        return ScenarioRun(
            scenario_id=scenario.id,
            scenario_title=scenario.title,
            tier=scenario.tier,
            backend_name=getattr(backend, "name", "?"),
            model_id=getattr(backend, "model_id", "?"),
            score=0.0,
            passed=False,
            elapsed_s=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )

    response = ResponseUnderTest(
        text=turn.text,
        tool_calls=list(turn.tool_calls_made),
        n_tool_iterations=turn.n_tool_iterations,
        corpus_doc_ids=doc_ids,
    )

    results = [run_grader(spec, response) for spec in scenario.graders]
    total_weight = sum(r.weight for r in results) or 1.0
    score = sum(r.weighted for r in results) / total_weight

    return ScenarioRun(
        scenario_id=scenario.id,
        scenario_title=scenario.title,
        tier=scenario.tier,
        backend_name=getattr(backend, "name", "?"),
        model_id=getattr(backend, "model_id", "?"),
        score=score,
        passed=score >= scenario.pass_threshold,
        grader_results=results,
        response_text=turn.text,
        tool_calls=list(turn.tool_calls_made),
        n_tool_iterations=turn.n_tool_iterations,
        elapsed_s=elapsed,
    )


def run_matrix(
    scenarios: list[Scenario],
    backends: list[Any],
    procedures_dir: str | Path,
    context: CopilotContext | None = None,
    system_prompt: str | None = None,
    progress: bool = True,
) -> list[ScenarioRun]:
    """Run every scenario against every backend.

    A fresh ToolRegistry is built per cell so tool-call state cannot leak
    between models — otherwise one model's retrieval could warm a cache
    the next one benefits from, which would quietly invalidate the
    comparison.
    """
    doc_ids = corpus_doc_ids(procedures_dir)
    rag_index = build_rag_index(procedures_dir)

    runs: list[ScenarioRun] = []
    total = len(scenarios) * len(backends)
    n = 0
    for backend in backends:
        for scenario in scenarios:
            n += 1
            if progress:
                label = getattr(backend, "model_id", getattr(backend, "name", "?"))
                print(f"[{n}/{total}] T{scenario.tier} {scenario.id} -> {label}", flush=True)

            ctx = CopilotContext(
                telemetry=None if context is None else context.telemetry,
                channel_names=[] if context is None else list(context.channel_names),
                sample_rate_hz=1.0 if context is None else context.sample_rate_hz,
                anomaly_scores=None if context is None else context.anomaly_scores,
                anomaly_threshold=None if context is None else context.anomaly_threshold,
                rag_index=rag_index,
                model_card_path=None if context is None else context.model_card_path,
            )
            tools = default_tool_registry(ctx)
            run = run_scenario(
                scenario, backend, tools, doc_ids, system_prompt=system_prompt
            )
            if progress:
                mark = "PASS" if run.passed else ("ERROR" if run.error else "fail")
                print(f"      {mark}  score={run.score:.2f}  {run.elapsed_s:.1f}s", flush=True)
            runs.append(run)
    return runs
