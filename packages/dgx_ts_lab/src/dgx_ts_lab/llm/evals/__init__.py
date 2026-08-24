"""Capability-escalation evaluation suite.

Answers one procurement question with evidence instead of opinion:
**at what point does a small local model stop being good enough?**

The suite runs graded operator scenarios across four escalating tiers
(Recall -> Synthesis -> Agentic -> Frontier) against any number of
`LLMBackend` implementations, and reports each model's *escalation
point* — the lowest tier where it starts failing.

See `configs/llm_evals/README.md` for how to add real-world scenarios.
"""

from .graders import GRADER_REGISTRY, GraderResult, ResponseUnderTest, run_grader
from .report import escalation_point, write_reports
from .runner import ScenarioRun, build_rag_index, corpus_doc_ids, run_matrix, run_scenario
from .scenario import TIER_NAMES, Scenario, load_scenario, load_scenario_dir

__all__ = [
    "GRADER_REGISTRY",
    "GraderResult",
    "ResponseUnderTest",
    "Scenario",
    "ScenarioRun",
    "TIER_NAMES",
    "build_rag_index",
    "corpus_doc_ids",
    "escalation_point",
    "load_scenario",
    "load_scenario_dir",
    "run_grader",
    "run_matrix",
    "run_scenario",
    "write_reports",
]
