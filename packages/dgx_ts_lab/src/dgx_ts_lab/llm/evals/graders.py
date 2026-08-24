"""Objective graders for capability-escalation scenarios.

Design principle
----------------
Every grader must be **machine-checkable**. A procurement demo that
rests on "the big model's answer read better" is worthless — the
reviewer discounts it as taste. These graders check facts that are true
or false independent of who is looking:

    * Did it call these tools, in this order?
    * Did it cite a document that actually exists in the corpus?
    * Is the number it produced within tolerance of the right answer?
    * Does its JSON validate against the required schema?
    * Did it acknowledge the constraint it was required to respect?

The most diagnostic grader here is ``no_hallucinated_citation``. Under
RAG pressure, small models invent plausible-looking document IDs
("per procedure EPS-114..."). That is a visible, damning failure mode
that needs no expert to interpret — the cited document simply does not
exist.

Each grader returns a GraderResult with a 0..1 score. Scenario score is
the weighted mean; the scenario passes if that meets its threshold.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraderResult:
    """Outcome of one grader against one model response."""

    grader: str
    score: float                     # 0.0 .. 1.0
    weight: float = 1.0
    detail: str = ""                 # human-readable "why this score"

    @property
    def weighted(self) -> float:
        return self.score * self.weight


@dataclass
class ResponseUnderTest:
    """Everything a grader may inspect about one model turn.

    Bundling this rather than passing loose args keeps grader signatures
    uniform, so the registry can dispatch generically.
    """

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    n_tool_iterations: int = 0
    corpus_doc_ids: set[str] = field(default_factory=set)   # for hallucination check

    @property
    def tool_names(self) -> list[str]:
        """Tool names in call order, duplicates preserved."""
        return [str(tc.get("name", "")) for tc in self.tool_calls]


# ── Individual graders ────────────────────────────────────────────────
#
# Signature: (response, config) -> (score, detail)
# `config` is the grader's YAML block minus `type` and `weight`.


def _grade_contains_all(r: ResponseUnderTest, cfg: dict) -> tuple[float, str]:
    """Fraction of required phrases present (case-insensitive)."""
    phrases: Sequence[str] = cfg.get("phrases", [])
    if not phrases:
        return 1.0, "no phrases configured"
    low = r.text.lower()
    hits = [p for p in phrases if p.lower() in low]
    missing = [p for p in phrases if p.lower() not in low]
    score = len(hits) / len(phrases)
    detail = f"{len(hits)}/{len(phrases)} present"
    if missing:
        detail += f"; missing: {missing}"
    return score, detail


def _grade_contains_any(r: ResponseUnderTest, cfg: dict) -> tuple[float, str]:
    """1.0 if any required phrase appears, else 0.0."""
    phrases: Sequence[str] = cfg.get("phrases", [])
    if not phrases:
        return 1.0, "no phrases configured"
    low = r.text.lower()
    hits = [p for p in phrases if p.lower() in low]
    return (1.0 if hits else 0.0), (f"matched: {hits}" if hits else f"none of {list(phrases)}")


def _grade_must_not_contain(r: ResponseUnderTest, cfg: dict) -> tuple[float, str]:
    """1.0 only if NO forbidden phrase appears.

    Used to catch confidently-wrong answers: e.g. a model that says
    "shed the comms transmitter first" when the eclipse precedence rule
    forbids it.
    """
    phrases: Sequence[str] = cfg.get("phrases", [])
    low = r.text.lower()
    bad = [p for p in phrases if p.lower() in low]
    return (0.0 if bad else 1.0), (f"FORBIDDEN present: {bad}" if bad else "clean")


def _grade_tool_sequence(r: ResponseUnderTest, cfg: dict) -> tuple[float, str]:
    """Score the agentic tool-call chain.

    `required`: tools that must be called.
    `ordered`:  if true, they must appear in the given relative order
                (other calls may interleave).
    `min_iterations`: minimum tool-loop round trips — catches models that
                fire everything blindly in one shot instead of using
                each result to decide the next call.
    """
    required: list[str] = list(cfg.get("required", []))
    ordered: bool = bool(cfg.get("ordered", False))
    min_iters: int = int(cfg.get("min_iterations", 0))

    called = r.tool_names
    if not required:
        return 1.0, "no tools required"

    present = [t for t in required if t in called]
    coverage = len(present) / len(required)
    details = [f"called {called}", f"coverage {len(present)}/{len(required)}"]

    order_ok = True
    if ordered and len(present) == len(required):
        # Walk `called` looking for `required` as a subsequence.
        pos = 0
        for want in required:
            try:
                pos = called.index(want, pos) + 1
            except ValueError:
                order_ok = False
                break
    elif ordered:
        order_ok = False

    iters_ok = r.n_tool_iterations >= min_iters
    if min_iters:
        details.append(f"iterations {r.n_tool_iterations} (need >= {min_iters})")

    score = coverage
    if ordered:
        score *= 1.0 if order_ok else 0.5
        details.append("order OK" if order_ok else "ORDER WRONG")
    if min_iters and not iters_ok:
        score *= 0.5
        details.append("TOO FEW ITERATIONS (fired tools blindly?)")

    return score, "; ".join(details)


_DOC_ID_RE = re.compile(r"\b([A-Z]{3})-(\d{3})\b")


def _grade_cites_documents(r: ResponseUnderTest, cfg: dict) -> tuple[float, str]:
    """Did the answer cite the documents it needed to?

    `required_all`: every one must be cited.
    `required_any`: at least one must be cited.
    Doc IDs are matched in the canonical ``ABC-123`` form.
    """
    cited = {f"{a}-{b}" for a, b in _DOC_ID_RE.findall(r.text)}
    req_all: list[str] = list(cfg.get("required_all", []))
    req_any: list[str] = list(cfg.get("required_any", []))

    if req_all:
        hits = [d for d in req_all if d in cited]
        score = len(hits) / len(req_all)
        missing = [d for d in req_all if d not in cited]
        return score, f"cited {sorted(cited)}; missing {missing}" if missing else f"cited all {req_all}"

    if req_any:
        hits = [d for d in req_any if d in cited]
        return (1.0 if hits else 0.0), f"cited {sorted(cited)}; wanted any of {req_any}"

    return (1.0 if cited else 0.0), f"cited {sorted(cited)}"


def _grade_no_hallucinated_citation(r: ResponseUnderTest, cfg: dict) -> tuple[float, str]:
    """Every ``ABC-123`` mentioned must exist in the corpus.

    The single most diagnostic check in the suite. Small models under
    RAG pressure fabricate document IDs; this makes that visible and
    non-arguable. Score is the fraction of citations that are real, so
    one bad cite among four costs 25%.
    """
    cited = {f"{a}-{b}" for a, b in _DOC_ID_RE.findall(r.text)}
    if not cited:
        # Nothing cited is not a hallucination. Citation *coverage* is
        # graded separately by cites_documents.
        return 1.0, "no document IDs cited"
    known = r.corpus_doc_ids
    if not known:
        return 1.0, "corpus doc IDs unavailable — check skipped"
    bogus = sorted(d for d in cited if d not in known)
    real = len(cited) - len(bogus)
    score = real / len(cited)
    if bogus:
        return score, f"HALLUCINATED: {bogus} (real: {real}/{len(cited)})"
    return 1.0, f"all {len(cited)} citations valid"


def _grade_numeric_answer(r: ResponseUnderTest, cfg: dict) -> tuple[float, str]:
    """Extract a number and compare against the expected value.

    `expected`:  target value
    `tolerance`: absolute tolerance (default 0)
    `unit_hint`: optional regex fragment that must sit near the number,
                 so "35 minutes" is not matched by "35 W".

    Scans all numbers in the text and passes if ANY is within tolerance —
    models narrate intermediate arithmetic, and demanding the final
    number be the only one present would fail correct answers.
    """
    expected = float(cfg["expected"])
    tol = float(cfg.get("tolerance", 0.0))
    unit = cfg.get("unit_hint")

    if unit:
        pattern = re.compile(rf"(-?\d+(?:\.\d+)?)\s*(?:{unit})", re.IGNORECASE)
        nums = [float(m) for m in pattern.findall(r.text)]
    else:
        nums = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", r.text)]

    if not nums:
        return 0.0, f"no number found (unit_hint={unit!r})"
    close = [n for n in nums if abs(n - expected) <= tol]
    if close:
        return 1.0, f"found {close[0]} within +/-{tol} of {expected}"
    nearest = min(nums, key=lambda n: abs(n - expected))
    return 0.0, f"nearest was {nearest}, expected {expected} +/-{tol}"


def _grade_json_schema(r: ResponseUnderTest, cfg: dict) -> tuple[float, str]:
    """Extract a JSON object and check required keys / types.

    Deliberately lightweight — checks presence of `required_keys` rather
    than running a full JSONSchema validator, so this has no extra dep.
    Structured-output failure is usually total (no JSON at all, or
    truncated), not subtle.
    """
    required_keys: list[str] = list(cfg.get("required_keys", []))
    blocks = re.findall(r"\{.*\}", r.text, flags=re.DOTALL)
    if not blocks:
        return 0.0, "no JSON object found in response"
    parsed = None
    for b in sorted(blocks, key=len, reverse=True):
        try:
            parsed = json.loads(b)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        return 0.0, "found brace-delimited text but it did not parse as JSON"
    if not isinstance(parsed, dict):
        return 0.0, f"parsed JSON was {type(parsed).__name__}, expected object"
    if not required_keys:
        return 1.0, "valid JSON object"
    hits = [k for k in required_keys if k in parsed]
    missing = [k for k in required_keys if k not in parsed]
    score = len(hits) / len(required_keys)
    return score, f"keys {len(hits)}/{len(required_keys)}" + (f"; missing {missing}" if missing else "")


def _grade_constraint_acknowledged(r: ResponseUnderTest, cfg: dict) -> tuple[float, str]:
    """Did the answer engage with each named constraint?

    Each constraint is a group of synonyms; the constraint counts as
    acknowledged if any synonym appears. Catches the classic small-model
    failure of optimizing one dimension while silently violating another.

    `constraints`: list of {name, any_of: [...]}
    """
    constraints: list[dict] = list(cfg.get("constraints", []))
    if not constraints:
        return 1.0, "no constraints configured"
    low = r.text.lower()
    met, unmet = [], []
    for c in constraints:
        name = str(c.get("name", "?"))
        syns = [s.lower() for s in c.get("any_of", [])]
        (met if any(s in low for s in syns) else unmet).append(name)
    score = len(met) / len(constraints)
    detail = f"addressed {met}"
    if unmet:
        detail += f"; IGNORED {unmet}"
    return score, detail


def _grade_min_length(r: ResponseUnderTest, cfg: dict) -> tuple[float, str]:
    """Guard against a model that 'passes' by emitting a one-liner.

    Not a quality measure on its own — only meaningful combined with
    substantive graders.
    """
    min_chars = int(cfg.get("min_chars", 200))
    n = len(r.text.strip())
    return (1.0 if n >= min_chars else n / min_chars), f"{n} chars (min {min_chars})"


GraderFn = Callable[[ResponseUnderTest, dict], tuple[float, str]]

GRADER_REGISTRY: dict[str, GraderFn] = {
    "contains_all": _grade_contains_all,
    "contains_any": _grade_contains_any,
    "must_not_contain": _grade_must_not_contain,
    "tool_sequence": _grade_tool_sequence,
    "cites_documents": _grade_cites_documents,
    "no_hallucinated_citation": _grade_no_hallucinated_citation,
    "numeric_answer": _grade_numeric_answer,
    "json_schema": _grade_json_schema,
    "constraint_acknowledged": _grade_constraint_acknowledged,
    "min_length": _grade_min_length,
}


def run_grader(spec: dict, response: ResponseUnderTest) -> GraderResult:
    """Dispatch one grader spec (a YAML block with `type`) against a response."""
    gtype = str(spec.get("type", "")).strip()
    fn = GRADER_REGISTRY.get(gtype)
    weight = float(spec.get("weight", 1.0))
    if fn is None:
        return GraderResult(
            grader=gtype or "<missing type>",
            score=0.0,
            weight=weight,
            detail=f"unknown grader type {gtype!r}; known: {sorted(GRADER_REGISTRY)}",
        )
    cfg = {k: v for k, v in spec.items() if k not in ("type", "weight")}
    score, detail = fn(response, cfg)
    return GraderResult(grader=gtype, score=max(0.0, min(1.0, score)), weight=weight, detail=detail)
