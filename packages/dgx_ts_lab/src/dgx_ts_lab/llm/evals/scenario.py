"""Scenario definitions for the capability-escalation suite.

A *scenario* is one operator request plus the objective graders that
decide whether a model handled it. Scenarios are grouped into tiers that
correspond to the capability escalation we are trying to demonstrate:

    Tier 1 — Recall.        Single-fact lookup. A 4B model should PASS.
                            This is the control group: it is what makes
                            the demo honest rather than rigged.
    Tier 2 — Synthesis.     Reconcile 2+ documents. Small models cite one
                            source and miss the conflict.
    Tier 3 — Agentic.       Chained tool calls where each depends on the
                            previous result. Small models fire one tool
                            or fabricate results.
    Tier 4 — Frontier.      Joint constraint satisfaction across power,
                            thermal, comms, and procedure precedence,
                            with arithmetic. This is where "a small local
                            model is good enough" stops being true.

The tier-1 control matters more than it looks. A suite where everything
fails on small models reads as rigged and gets discounted. A suite that
says "yes, the 4B handles basic recall correctly — and here is the exact
point where it falls off" is far harder to argue with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TIER_NAMES = {
    1: "Recall",
    2: "Synthesis",
    3: "Agentic",
    4: "Frontier",
}


@dataclass
class Scenario:
    """One graded operator request.

    Attributes
    ----------
    id
        Stable slug, used as the results-matrix row key.
    tier
        1-4, see module docstring.
    title
        Short human label for the report.
    why_it_escalates
        Prose explaining what capability this probes and how a weaker
        model is expected to fail. This lands in the report so a reader
        who is not an ML engineer understands what they are looking at.
    prompt
        The operator's message, sent verbatim to the co-pilot.
    graders
        List of grader specs (see graders.GRADER_REGISTRY).
    pass_threshold
        Weighted score at or above which the scenario counts as passed.
    max_tool_iters
        Tool-loop budget for this scenario. Tier 3/4 need more.
    expected_min_tier
        Which model tier we expect to be the first to pass this. Purely
        documentation — the run reports what actually happened.
    """

    id: str
    tier: int
    title: str
    prompt: str
    graders: list[dict[str, Any]] = field(default_factory=list)
    why_it_escalates: str = ""
    pass_threshold: float = 0.7
    max_tool_iters: int = 6
    expected_min_tier: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def tier_name(self) -> str:
        return TIER_NAMES.get(self.tier, f"Tier{self.tier}")

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the scenario is well-formed.

        Checked at load time so a typo in a YAML surfaces immediately
        rather than as a mysterious zero score after an expensive run.
        """
        from .graders import GRADER_REGISTRY

        problems: list[str] = []
        if not self.id:
            problems.append("missing id")
        if self.tier not in TIER_NAMES:
            problems.append(f"tier {self.tier} not in {sorted(TIER_NAMES)}")
        if not self.prompt.strip():
            problems.append("empty prompt")
        if not self.graders:
            problems.append("no graders — scenario would always score 0")
        for i, g in enumerate(self.graders):
            gtype = g.get("type")
            if gtype not in GRADER_REGISTRY:
                problems.append(
                    f"grader[{i}] type {gtype!r} unknown; "
                    f"known: {sorted(GRADER_REGISTRY)}"
                )
        if not 0.0 < self.pass_threshold <= 1.0:
            problems.append(f"pass_threshold {self.pass_threshold} outside (0, 1]")
        return problems


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate one scenario YAML."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenario = Scenario(
        id=str(raw.get("id", path.stem)),
        tier=int(raw.get("tier", 1)),
        title=str(raw.get("title", path.stem)),
        prompt=str(raw.get("prompt", "")),
        graders=list(raw.get("graders", [])),
        why_it_escalates=str(raw.get("why_it_escalates", "")).strip(),
        pass_threshold=float(raw.get("pass_threshold", 0.7)),
        max_tool_iters=int(raw.get("max_tool_iters", 6)),
        expected_min_tier=str(raw.get("expected_min_tier", "")),
        tags=list(raw.get("tags", [])),
    )
    problems = scenario.validate()
    if problems:
        raise ValueError(f"invalid scenario {path}:\n  - " + "\n  - ".join(problems))
    return scenario


def load_scenario_dir(root: str | Path, tiers: list[int] | None = None) -> list[Scenario]:
    """Load every ``*.yaml`` under ``root`` recursively, sorted by (tier, id).

    ``tiers`` optionally filters to a subset, e.g. ``[3, 4]`` to run only
    the scenarios that actually discriminate between model sizes.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"scenario directory not found: {root}")
    scenarios = [
        load_scenario(p)
        for p in sorted(root.rglob("*.yaml"))
        if p.name.lower() != "readme.yaml"
    ]
    if tiers:
        wanted = set(tiers)
        scenarios = [s for s in scenarios if s.tier in wanted]
    return sorted(scenarios, key=lambda s: (s.tier, s.id))
