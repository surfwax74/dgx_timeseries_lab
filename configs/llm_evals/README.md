# `configs/llm_evals/` — Capability Escalation Suite

Graded operator scenarios that answer one question with evidence rather
than opinion:

> **At what point does a small local model stop being good enough?**

This exists because basic Q&A demos *lose* the procurement argument. Ask
a 4B model "what does a battery SoC anomaly indicate?" and it answers
fine, and the reviewer correctly concludes that a small local model is
sufficient. The suite is built to find and document the exact point
where that stops being true.

## Running it

```bash
# Smoke, no GPU, no network
python scripts/run_capability_evals.py --backend mock

# See what would run without calling any model
python scripts/run_capability_evals.py --list

# The real comparison: small local vs. DGX-hosted frontier.
# Serve each model on its own port first.
python scripts/run_capability_evals.py \
    --model ollama:gemma4:e4b \
    --model vllm:google/gemma-4-26B-A4B-it@http://localhost:8000/v1 \
    --model vllm:meta-llama/Llama-3.1-405B-Instruct@http://localhost:8001/v1 \
    --output runs/capability_evals

# Only the tiers that discriminate (skip the tier-1 control)
python scripts/run_capability_evals.py --model ... --tiers 3,4
```

Output lands in `runs/capability_evals/`:
- `capability_report.md` — escalation table, pass matrix, per-scenario detail
- `capability_report.json` — same data, machine-readable

## The four tiers

| Tier | Name | What it probes | Expected to pass |
|---|---|---|---|
| 1 | Recall | One fact, one document | Everything, incl. 4B |
| 2 | Synthesis | Reconcile 2+ documents that disagree | 12B and up |
| 3 | Agentic | Chained tool calls, each depending on the last | 26B MoE / 70B |
| 4 | Frontier | Joint constraints + arithmetic + precedence | 70B / 405B |

**Tier 1 is deliberately easy and must pass everywhere.** It is the
control group. A suite where the small model fails everything reads as
rigged and gets discounted by exactly the audience you are trying to
convince. "The 4B handles basic recall correctly, and here is precisely
where it falls off" is a much harder claim to wave away.

## Why the grading is objective

Every grader is machine-checkable. None of them ask whether an answer
"read well":

| Grader | Checks |
|---|---|
| `contains_all` / `contains_any` | Required phrases present |
| `must_not_contain` | Forbidden phrase absent |
| `cites_documents` | Specific `ABC-123` doc IDs cited |
| `no_hallucinated_citation` | **Every cited doc ID actually exists** |
| `tool_sequence` | Required tools called, optionally in order, with a minimum number of loop iterations |
| `numeric_answer` | A number within tolerance, optionally near a unit |
| `json_schema` | Output parses as JSON and has required keys |
| `constraint_acknowledged` | Each named constraint was engaged with |
| `min_length` | Guards against passing via a one-liner |

`no_hallucinated_citation` is the most diagnostic of these. Under RAG
pressure, small models invent plausible document IDs — "per procedure
EPS-114..." — and EPS-114 does not exist. That failure needs no domain
expert to adjudicate, which makes it ideal demo evidence.

`tool_sequence`'s `min_iterations` catches the other characteristic
failure: a model that fires every tool at once in a single iteration
with guessed arguments, rather than using each result to decide the next
call. Same tools called, no actual reasoning chain.

## The RAG corpus

Scenarios retrieve against `docs/procedures/` — six interlocking
procedure documents built so that multi-hop reasoning is genuinely
required:

| Doc | Role |
|---|---|
| `SYS-000` | Procedure precedence — resolves conflicts between the others |
| `EPS-201` | Undervoltage response — shed order A |
| `SYS-001` | Safe mode entry — shed order B, **contradicts EPS-201** |
| `TCS-105` | Thermal limits, margins, heater cooling rates |
| `EPS-310` | Power budget with a 1.2 recovery-margin factor |
| `COM-220` | Ground contact windows, command-loss timer |

The EPS-201 / SYS-001 contradiction is deliberate and is resolvable only
via SYS-000's precedence rules. That single design decision is what
makes tiers 2 and 4 hard: no individual document contains the answer.

## Adding your own scenarios

This is the part meant for the team. Real work examples will
discriminate better than anything generic, because they encode
constraints only your operators know.

### 1. Add your documents

Drop procedure markdown into `docs/procedures/` (or point
`--procedures` at your own directory). Filenames must start with a
`ABC-123` ID so the hallucination grader can tell real citations from
invented ones:

```
docs/procedures/ADC-410_wheel_desaturation.md
```

Sanitize before committing — this corpus goes in git.

### 2. Write the scenario

Create `configs/llm_evals/tier<N>_<name>/<id>.yaml`, or start a
separate directory and pass both:

```bash
python scripts/run_capability_evals.py \
    --scenarios configs/llm_evals \
    --scenarios configs/llm_evals_team
```

```yaml
id: t4_wheel_saturation_recovery      # unique; becomes the report row key
tier: 4                                # 1-4
title: "Reaction wheel saturation during a mandatory contact"
expected_min_tier: "70B / 405B"        # documentation only

why_it_escalates: >
  Explain what capability this probes and how a weaker model is
  expected to fail. This text goes into the report, so write it for a
  reader who is not an ML engineer — it is what makes the chart
  interpretable.

prompt: |
  The operator's actual message. Include concrete numbers so the
  graders have something to check.

max_tool_iters: 10
pass_threshold: 0.7

graders:
  - type: cites_documents
    required_all: ["ADC-410", "COM-220"]
    weight: 0.4
  - type: no_hallucinated_citation
    weight: 0.2
  - type: numeric_answer
    expected: 18
    tolerance: 0.5
    unit_hint: "W|watt"
    weight: 0.4
```

Weights are relative; the score is their weighted mean.

### 3. Validate before you spend GPU time

```bash
python scripts/run_capability_evals.py --list
```

This loads and validates every scenario without calling a model. Bad
grader names, empty prompts, and out-of-range thresholds fail here
rather than after an hour of inference.

### 4. Prove the scenario actually discriminates

**Do this.** A scenario is only useful if a correct answer passes *and*
a naive answer fails. Add both cases to
`packages/dgx_ts_lab/tests/test_capability_evals.py`, following the
existing `test_gold_response_passes_*` and
`test_naive_wrong_answer_fails_*` pattern:

```python
def test_gold_response_passes_my_scenario() -> None:
    s = load_scenario(SCENARIO_DIR / "tier4_frontier" / "my_scenario.yaml")
    gold = _r("...the answer a competent model gives...")
    assert _score(s, gold) >= s.pass_threshold


def test_naive_answer_fails_my_scenario() -> None:
    s = load_scenario(SCENARIO_DIR / "tier4_frontier" / "my_scenario.yaml")
    naive = _r("...the plausible-but-wrong answer...")
    assert _score(s, naive) < s.pass_threshold
```

Without the second test you may ship a scenario nothing can pass, which
produces a chart that looks damning and is actually just broken. Without
the first, you may ship one everything passes, which is noise.

## Wiring in live telemetry

By default the tool-calling scenarios run with no telemetry loaded, so
`query_telemetry` returns an error string. The model still has to *call*
it, which is what `tool_sequence` grades — fine for measuring agentic
behavior.

For a demo where the answers reference real numbers, build a
`CopilotContext` with telemetry loaded and pass it to `run_matrix()`.
See `scripts/dgx_showcase_copilot_qna.py` for the pattern.

## Reading the report

The headline is the **escalation point** per model: the lowest tier
where it first fails. Not an average across tiers — averaging blurs
exactly the boundary the report exists to locate.

A model whose escalation point is Tier 2 cannot be trusted with
multi-document reasoning. One that only falls off at Tier 4 is failing
joint constraint satisfaction, which is what real anomaly response
actually is.

## Related

- Serving the models under test: [`docs/frontier_model_serving.md`](../../docs/frontier_model_serving.md)
- Backend inventory and tier picker: [`configs/llm/README.md`](../llm/README.md)
- Co-pilot internals: `packages/dgx_ts_lab/src/dgx_ts_lab/llm/README.md`
