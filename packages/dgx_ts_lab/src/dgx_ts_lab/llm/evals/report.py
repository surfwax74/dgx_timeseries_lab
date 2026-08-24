"""Render capability-escalation results as Markdown + JSON.

The report is the deliverable. Its job is to answer one question for a
reader who is not an ML engineer:

    "At what point does a small local model stop being good enough?"

So the headline is the **escalation point** per model — the lowest tier
where that model starts failing — not an aggregate score. An average
across tiers would blur exactly the boundary we are trying to show.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .runner import ScenarioRun
from .scenario import TIER_NAMES, Scenario


def _model_key(run: ScenarioRun) -> str:
    return run.model_id or run.backend_name


def escalation_point(runs: list[ScenarioRun], model: str) -> tuple[int | None, str]:
    """Lowest tier at which `model` failed at least one scenario.

    Returns (tier, explanation). ``None`` means the model passed every
    tier it was given — the honest answer being "this suite did not find
    its ceiling", not "it is infinitely capable".
    """
    by_tier: dict[int, list[ScenarioRun]] = defaultdict(list)
    for r in runs:
        if _model_key(r) == model:
            by_tier[r.tier].append(r)

    for tier in sorted(by_tier):
        tier_runs = by_tier[tier]
        failures = [r for r in tier_runs if not r.passed]
        if failures:
            names = ", ".join(f.scenario_id for f in failures)
            return tier, f"first failed at Tier {tier} ({TIER_NAMES.get(tier, '?')}): {names}"
    return None, "passed every tier in this suite"


def write_reports(
    runs: list[ScenarioRun],
    scenarios: list[Scenario],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write ``capability_report.md`` and ``capability_report.json``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    models = sorted({_model_key(r) for r in runs})
    cell: dict[tuple[str, str], ScenarioRun] = {
        (r.scenario_id, _model_key(r)): r for r in runs
    }

    lines: list[str] = []
    lines.append("# Capability Escalation Report")
    lines.append("")
    lines.append(
        "Each scenario is graded by objective, machine-checkable criteria — "
        "tool-call sequences, citation validity, numeric tolerance, schema "
        "conformance. No subjective quality judgment is involved, so these "
        "results do not depend on who is reading them."
    )
    lines.append("")

    # ── Headline: where each model falls off ─────────────────────────
    lines.append("## Escalation point — where each model stops being sufficient")
    lines.append("")
    lines.append("| Model | Escalation point | Detail |")
    lines.append("|---|---|---|")
    for m in models:
        tier, why = escalation_point(runs, m)
        label = f"**Tier {tier} — {TIER_NAMES.get(tier, '?')}**" if tier else "_none found_"
        lines.append(f"| `{m}` | {label} | {why} |")
    lines.append("")
    lines.append(
        "> A model whose escalation point is Tier 1 or 2 cannot be trusted "
        "with multi-document reasoning. A model that only falls off at "
        "Tier 4 is failing joint constraint satisfaction — the class of "
        "problem that real anomaly response actually is."
    )
    lines.append("")

    # ── Pass matrix ──────────────────────────────────────────────────
    lines.append("## Pass matrix")
    lines.append("")
    header = "| Tier | Scenario | " + " | ".join(f"`{m}`" for m in models) + " |"
    sep = "|---|---|" + "|".join([":---:"] * len(models)) + "|"
    lines.append(header)
    lines.append(sep)
    for s in sorted(scenarios, key=lambda x: (x.tier, x.id)):
        cells = []
        for m in models:
            r = cell.get((s.id, m))
            if r is None:
                cells.append("—")
            elif r.error:
                cells.append("⚠️")
            else:
                cells.append(f"✅ {r.score:.2f}" if r.passed else f"❌ {r.score:.2f}")
        lines.append(f"| T{s.tier} | {s.title} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("✅ pass · ❌ fail · ⚠️ backend error · — not run")
    lines.append("")

    # ── Per-tier summary ─────────────────────────────────────────────
    lines.append("## Pass rate by tier")
    lines.append("")
    lines.append("| Tier | " + " | ".join(f"`{m}`" for m in models) + " |")
    lines.append("|---|" + "|".join([":---:"] * len(models)) + "|")
    for tier in sorted({s.tier for s in scenarios}):
        row = [f"T{tier} — {TIER_NAMES.get(tier, '?')}"]
        for m in models:
            tr = [r for r in runs if r.tier == tier and _model_key(r) == m]
            if not tr:
                row.append("—")
                continue
            n_pass = sum(1 for r in tr if r.passed)
            row.append(f"{n_pass}/{len(tr)}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Scenario detail ──────────────────────────────────────────────
    lines.append("## Scenario detail")
    lines.append("")
    for s in sorted(scenarios, key=lambda x: (x.tier, x.id)):
        lines.append(f"### T{s.tier} · {s.title}")
        lines.append("")
        lines.append(f"`{s.id}` — pass threshold {s.pass_threshold:.2f}")
        lines.append("")
        if s.why_it_escalates:
            lines.append(f"**Why this escalates.** {s.why_it_escalates}")
            lines.append("")
        lines.append("<details><summary>Prompt</summary>")
        lines.append("")
        lines.append("```text")
        lines.append(s.prompt.strip())
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")
        for m in models:
            r = cell.get((s.id, m))
            if r is None:
                continue
            status = "ERROR" if r.error else ("PASS" if r.passed else "FAIL")
            lines.append(f"**`{m}` — {status} ({r.score:.2f})**")
            lines.append("")
            if r.error:
                lines.append(f"- backend error: `{r.error}`")
                lines.append("")
                continue
            lines.append(
                f"- tools called: `{[tc.get('name') for tc in r.tool_calls]}` "
                f"in {r.n_tool_iterations} iteration(s), {r.elapsed_s:.1f}s"
            )
            for g in r.grader_results:
                mark = "✅" if g.score >= 0.999 else ("⚠️" if g.score > 0 else "❌")
                lines.append(f"- {mark} `{g.grader}` {g.score:.2f} — {g.detail}")
            lines.append("")
        lines.append("")

    md_path = out / "capability_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "models": models,
        "escalation_points": {
            m: {"tier": escalation_point(runs, m)[0], "detail": escalation_point(runs, m)[1]}
            for m in models
        },
        "scenarios": [
            {
                "id": s.id, "tier": s.tier, "title": s.title,
                "pass_threshold": s.pass_threshold,
                "why_it_escalates": s.why_it_escalates,
            }
            for s in scenarios
        ],
        "runs": [
            {
                **{k: v for k, v in asdict(r).items() if k != "grader_results"},
                "grader_results": [asdict(g) for g in r.grader_results],
            }
            for r in runs
        ],
    }
    json_path = out / "capability_report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {"markdown": md_path, "json": json_path}
