"""Tests for the capability-escalation eval suite.

The most important tests here are the "gold response" ones. A grading
suite that no model can pass is broken, not discriminating — so for
each shipped tier we construct the answer a competent model would give
and assert it clears the threshold. Without this, a typo in a grader
config would silently make the suite look like every model failed, and
the resulting procurement chart would be a lie.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dgx_ts_lab.llm.evals import (
    GRADER_REGISTRY,
    ResponseUnderTest,
    escalation_point,
    load_scenario,
    load_scenario_dir,
    run_grader,
)
from dgx_ts_lab.llm.evals.runner import ScenarioRun, corpus_doc_ids

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = REPO_ROOT / "configs" / "llm_evals"
PROCEDURES_DIR = REPO_ROOT / "docs" / "procedures"

REAL_DOC_IDS = {"SYS-000", "SYS-001", "EPS-201", "EPS-310", "TCS-105", "COM-220"}


# ── Corpus + scenario loading ─────────────────────────────────────────


def test_procedures_corpus_exists_with_expected_ids() -> None:
    ids = corpus_doc_ids(PROCEDURES_DIR)
    assert ids == REAL_DOC_IDS, f"corpus doc IDs drifted: {ids}"


def test_all_shipped_scenarios_load_and_validate() -> None:
    scenarios = load_scenario_dir(SCENARIO_DIR)
    assert len(scenarios) >= 9
    for s in scenarios:
        assert s.validate() == [], f"{s.id}: {s.validate()}"


def test_every_tier_is_represented() -> None:
    tiers = {s.tier for s in load_scenario_dir(SCENARIO_DIR)}
    assert tiers == {1, 2, 3, 4}, f"expected all four tiers, got {tiers}"


def test_tier_filter_selects_subset() -> None:
    only_34 = load_scenario_dir(SCENARIO_DIR, tiers=[3, 4])
    assert only_34
    assert {s.tier for s in only_34} == {3, 4}


def test_scenario_ids_are_unique() -> None:
    ids = [s.id for s in load_scenario_dir(SCENARIO_DIR)]
    assert len(ids) == len(set(ids)), "duplicate scenario ids"


def test_invalid_scenario_raises_with_detail(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "id: bad\ntier: 9\ntitle: nope\nprompt: ''\ngraders: []\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid scenario"):
        load_scenario(bad)


def test_unknown_grader_type_is_rejected_at_load(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "id: b\ntier: 1\ntitle: t\nprompt: hi\n"
        "graders:\n  - type: does_not_exist\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does_not_exist"):
        load_scenario(bad)


# ── Individual graders ────────────────────────────────────────────────


def _r(text: str, tool_calls=None, iters: int = 0) -> ResponseUnderTest:
    return ResponseUnderTest(
        text=text,
        tool_calls=tool_calls or [],
        n_tool_iterations=iters,
        corpus_doc_ids=set(REAL_DOC_IDS),
    )


def test_contains_all_partial_credit() -> None:
    res = run_grader(
        {"type": "contains_all", "phrases": ["alpha", "beta", "gamma"]},
        _r("alpha and beta only"),
    )
    assert res.score == pytest.approx(2 / 3)


def test_must_not_contain_catches_forbidden() -> None:
    ok = run_grader({"type": "must_not_contain", "phrases": ["danger"]}, _r("all clear"))
    bad = run_grader({"type": "must_not_contain", "phrases": ["danger"]}, _r("DANGER here"))
    assert ok.score == 1.0
    assert bad.score == 0.0


def test_no_hallucinated_citation_flags_fake_doc() -> None:
    good = run_grader({"type": "no_hallucinated_citation"}, _r("per EPS-201 and TCS-105"))
    bad = run_grader({"type": "no_hallucinated_citation"}, _r("per EPS-114 and EPS-201"))
    assert good.score == 1.0
    assert bad.score == pytest.approx(0.5)
    assert "EPS-114" in bad.detail


def test_no_hallucinated_citation_neutral_when_nothing_cited() -> None:
    res = run_grader({"type": "no_hallucinated_citation"}, _r("no ids here"))
    assert res.score == 1.0


def test_cites_documents_required_all() -> None:
    res = run_grader(
        {"type": "cites_documents", "required_all": ["EPS-201", "TCS-105"]},
        _r("only EPS-201 mentioned"),
    )
    assert res.score == pytest.approx(0.5)


def test_tool_sequence_order_and_iterations() -> None:
    calls = [{"name": "query_telemetry"}, {"name": "query_anomaly_history"}, {"name": "lookup_procedure"}]
    good = run_grader(
        {
            "type": "tool_sequence",
            "required": ["query_telemetry", "query_anomaly_history", "lookup_procedure"],
            "ordered": True,
            "min_iterations": 2,
        },
        _r("x", tool_calls=calls, iters=3),
    )
    assert good.score == pytest.approx(1.0)

    # Same tools, wrong order -> penalized.
    reversed_calls = list(reversed(calls))
    wrong_order = run_grader(
        {
            "type": "tool_sequence",
            "required": ["query_telemetry", "query_anomaly_history", "lookup_procedure"],
            "ordered": True,
        },
        _r("x", tool_calls=reversed_calls, iters=3),
    )
    assert wrong_order.score < 1.0


def test_tool_sequence_penalizes_blind_single_shot() -> None:
    """All tools fired in one iteration = guessed args, not reasoning."""
    calls = [{"name": "query_telemetry"}, {"name": "query_anomaly_history"}]
    res = run_grader(
        {
            "type": "tool_sequence",
            "required": ["query_telemetry", "query_anomaly_history"],
            "min_iterations": 2,
        },
        _r("x", tool_calls=calls, iters=1),
    )
    assert res.score < 1.0
    assert "TOO FEW ITERATIONS" in res.detail


def test_numeric_answer_with_unit_hint_discriminates() -> None:
    # "35 W" must not satisfy a query for 35 minutes.
    res = run_grader(
        {"type": "numeric_answer", "expected": 35, "tolerance": 1, "unit_hint": "min|minute"},
        _r("the load is 35 W"),
    )
    assert res.score == 0.0
    ok = run_grader(
        {"type": "numeric_answer", "expected": 35, "tolerance": 1, "unit_hint": "min|minute"},
        _r("about 35 minutes remain"),
    )
    assert ok.score == 1.0


def test_numeric_answer_tolerates_narrated_arithmetic() -> None:
    """Intermediate numbers in the working must not cause a false fail."""
    res = run_grader(
        {"type": "numeric_answer", "expected": 18, "tolerance": 0.5, "unit_hint": "W|watt"},
        _r("deficit is 15 W, times 1.2 gives 18 W required"),
    )
    assert res.score == 1.0


def test_json_schema_detects_missing_keys() -> None:
    res = run_grader(
        {"type": "json_schema", "required_keys": ["a", "b", "c", "d"]},
        _r('prose then {"a": 1, "b": 2}'),
    )
    assert res.score == pytest.approx(0.5)


def test_json_schema_zero_when_absent() -> None:
    res = run_grader({"type": "json_schema", "required_keys": ["a"]}, _r("no json at all"))
    assert res.score == 0.0


def test_constraint_acknowledged_reports_ignored() -> None:
    res = run_grader(
        {
            "type": "constraint_acknowledged",
            "constraints": [
                {"name": "thermal", "any_of": ["thermal margin"]},
                {"name": "comms", "any_of": ["ground contact"]},
            ],
        },
        _r("we considered the thermal margin only"),
    )
    assert res.score == pytest.approx(0.5)
    assert "IGNORED" in res.detail and "comms" in res.detail


def test_unknown_grader_scores_zero_not_crash() -> None:
    res = run_grader({"type": "nonexistent"}, _r("x"))
    assert res.score == 0.0
    assert "unknown grader" in res.detail


# ── Gold responses: the suite must be passable ────────────────────────
#
# For each shipped tier, construct the answer a competent model would
# produce and assert it clears the scenario threshold.


def _score(scenario, response: ResponseUnderTest) -> float:
    results = [run_grader(spec, response) for spec in scenario.graders]
    total_w = sum(r.weight for r in results) or 1.0
    return sum(r.weighted for r in results) / total_w


def test_gold_response_passes_tier1() -> None:
    s = load_scenario(SCENARIO_DIR / "tier1_recall" / "t1_battery_thermal_limit.yaml")
    gold = _r(
        "The battery pack operational maximum is +40 °C, specified in "
        "TCS-105 (Thermal Limits and Margins). Its survival maximum is +55 °C."
    )
    assert _score(s, gold) >= s.pass_threshold


def test_gold_response_passes_tier2_conflict() -> None:
    s = load_scenario(SCENARIO_DIR / "tier2_synthesis" / "t2_shed_order_conflict.yaml")
    gold = _r(
        "EPS-201 and SYS-001 do conflict on shed ordering. SYS-000 Rev C rule 3 "
        "resolves it: during eclipse, EPS-201 takes precedence over SYS-001. "
        "Since we are in eclipse, use the EPS-201 order: payload first."
    )
    assert _score(s, gold) >= s.pass_threshold


def test_gold_response_passes_tier3_agentic() -> None:
    s = load_scenario(SCENARIO_DIR / "tier3_agentic" / "t3_anomaly_triage_chain.yaml")
    gold = _r(
        "Step 1: queried bus_voltage — mean 27.9 V, last value 26.2 V, a clear "
        "decline. Step 2: queried the anomaly history for the same window — a "
        "detection fired at high score. Step 3: retrieved EPS-201, the EPS Bus "
        "Undervoltage Response procedure, which triggers below 26.5 V. The "
        "recommended first action is to shed the payload and observe for 60 s "
        "before proceeding to the next shed step.",
        tool_calls=[
            {"name": "query_telemetry"},
            {"name": "query_anomaly_history"},
            {"name": "lookup_procedure"},
        ],
        iters=3,
    )
    assert _score(s, gold) >= s.pass_threshold


def test_gold_response_passes_tier4_flagship() -> None:
    """The hardest shipped scenario must be reachable by a correct answer."""
    s = load_scenario(
        SCENARIO_DIR / "tier4_frontier" / "t4_eclipse_recovery_conflict.yaml"
    )
    gold = _r(
        "Bus voltage 26.1 V is below the 26.5 V trigger in EPS-201, so the "
        "undervoltage procedure is active. However I am NOT following its shed "
        "order literally, for two reasons.\n\n"
        "First, the battery pack is at -2 °C against a -5 °C operational "
        "minimum per TCS-105 — a thermal margin of 3 °C, under the 5 °C "
        "threshold. SYS-000 Rev C rule 2 therefore elevates thermal above EPS "
        "load management, so the secondary heaters must be retained.\n\n"
        "Second, the mandatory ground contact is 30 minutes out, inside the "
        "45-minute exclusion in EPS-201 step 3 and covered by COM-220. "
        "SYS-000 rule 4 forbids shedding the comms transmitter.\n\n"
        "Permissible sheds are therefore the payload (12 W) and reaction wheel "
        "desaturation (6 W), totalling 18 W. I will shed the payload first, "
        "wait 60 s, then desaturation if the decline continues.",
        iters=4,
    )
    assert _score(s, gold) >= s.pass_threshold


def test_gold_response_passes_tier4_arithmetic() -> None:
    s = load_scenario(
        SCENARIO_DIR / "tier4_frontier" / "t4_shed_budget_arithmetic.yaml"
    )
    gold = _r(
        "Per EPS-310, nominal draw is 55 W and generation has fallen to 40 W, "
        "so the deficit is 15 W. Applying the 1.2 recovery margin factor gives "
        "a required shed of 18 W. From EPS-201 that is the payload (12 W) plus "
        "reaction wheel desaturation (6 W)."
    )
    assert _score(s, gold) >= s.pass_threshold


def test_naive_wrong_answer_fails_tier4_flagship() -> None:
    """The characteristic small-model answer must actually score below threshold.

    This is the other half of suite validity: if the naive answer passed,
    the scenario would not discriminate and the chart would be noise.
    """
    s = load_scenario(
        SCENARIO_DIR / "tier4_frontier" / "t4_eclipse_recovery_conflict.yaml"
    )
    naive = _r(
        "Per EPS-201, bus voltage below 26.5 V triggers the undervoltage "
        "response. Shed in order: payload first, then secondary heaters, then "
        "the comms transmitter, then reaction wheel desaturation."
    )
    assert _score(s, naive) < s.pass_threshold


def test_hallucinating_answer_fails_tier3() -> None:
    s = load_scenario(SCENARIO_DIR / "tier3_agentic" / "t3_grounded_next_steps.yaml")
    fabricated = _r(
        "Follow procedure EPS-114 section 3 and then PWR-902 for the recovery "
        "sequence. Shed the payload per EPS-114.",
        tool_calls=[],
        iters=0,
    )
    assert _score(s, fabricated) < s.pass_threshold


# ── Report helpers ────────────────────────────────────────────────────


def _run(sid: str, tier: int, model: str, passed: bool) -> ScenarioRun:
    return ScenarioRun(
        scenario_id=sid, scenario_title=sid, tier=tier,
        backend_name="b", model_id=model,
        score=1.0 if passed else 0.0, passed=passed,
    )


def test_escalation_point_finds_lowest_failing_tier() -> None:
    runs = [
        _run("a", 1, "m", True),
        _run("b", 2, "m", True),
        _run("c", 3, "m", False),
        _run("d", 4, "m", False),
    ]
    tier, why = escalation_point(runs, "m")
    assert tier == 3
    assert "Tier 3" in why


def test_escalation_point_none_when_all_pass() -> None:
    runs = [_run("a", 1, "m", True), _run("b", 4, "m", True)]
    tier, why = escalation_point(runs, "m")
    assert tier is None
    assert "passed every tier" in why


def test_write_reports_emits_both_files(tmp_path: Path) -> None:
    from dgx_ts_lab.llm.evals import write_reports

    scenarios = load_scenario_dir(SCENARIO_DIR, tiers=[1])
    runs = [_run(s.id, s.tier, "m", True) for s in scenarios]
    paths = write_reports(runs, scenarios, tmp_path)
    assert paths["markdown"].exists()
    assert paths["json"].exists()
    md = paths["markdown"].read_text(encoding="utf-8")
    assert "Escalation point" in md
    assert "Pass matrix" in md


def test_grader_registry_is_documented() -> None:
    """Every registered grader should be reachable by name from YAML."""
    assert set(GRADER_REGISTRY) >= {
        "contains_all", "contains_any", "must_not_contain", "tool_sequence",
        "cites_documents", "no_hallucinated_citation", "numeric_answer",
        "json_schema", "constraint_acknowledged", "min_length",
    }
