"""B8 — Natural language → validated command sequence.

Operator says "spin up the payload but skip the eclipse warmup", and we
ask the LLM to produce a command sequence that:

    1. uses only opcodes from the known vocabulary (CommandTokenizer)
    2. validates against a simulator/validator that returns errors
    3. retries on validation failure (up to ``max_validation_iters``)

The simulator is a callback the caller supplies — Phase 8's cyber
command sequence world makes a natural validator.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from .backend import GenerateOptions, LLMBackend, SystemMessage, UserMessage


@dataclass
class CommandStep:
    """One step in a synthesized command sequence."""

    opcode: str
    params: dict[str, str] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {"opcode": self.opcode, "params": self.params, "rationale": self.rationale}


@dataclass
class ProcedureSynthResult:
    """Validated command sequence + audit trail."""

    success: bool
    steps: list[CommandStep] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    n_iterations: int = 0
    final_prompt: str = ""
    raw_text: str = ""


CommandValidator = Callable[[list[CommandStep]], list[str]]
"""Validator returns a (possibly empty) list of error strings. Empty = OK."""


_SYNTH_SYSTEM = (
    "You are a satellite command-sequence planner. Given a natural-language "
    "request and a list of available opcodes, produce a JSON-only response "
    "of the form: {\"steps\": [{\"opcode\": \"NAME\", \"params\": {\"k\": \"v\"}, "
    "\"rationale\": \"why\"}, ...]}. Do NOT include any other text. Only use "
    "opcodes from the provided vocabulary. Keep sequences short and ordered."
)


def _user_prompt(request: str, opcodes: list[str], params: list[str], errors: list[str]) -> str:
    parts = [
        f"Request: {request}",
        f"Available opcodes: {', '.join(opcodes)}",
        f"Common parameter values: {', '.join(params)}",
    ]
    if errors:
        parts.append(
            "Your previous attempt had these validation errors. Fix them:\n"
            + "\n".join(f"- {e}" for e in errors)
        )
    return "\n\n".join(parts)


def _parse_steps(raw: str) -> tuple[list[CommandStep], str | None]:
    """Returns (steps, parse_error_or_None)."""
    text = raw.strip()
    # Some models wrap JSON in ```json fences — strip if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
        if text.startswith("json"):
            text = text[4:].lstrip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return [], f"could not parse JSON: {e}"
    raw_steps = obj.get("steps") if isinstance(obj, dict) else None
    if not isinstance(raw_steps, list):
        return [], "no 'steps' list in response"
    steps: list[CommandStep] = []
    for i, s in enumerate(raw_steps):
        if not isinstance(s, dict) or "opcode" not in s:
            return [], f"step {i} missing 'opcode'"
        steps.append(
            CommandStep(
                opcode=str(s["opcode"]),
                params={str(k): str(v) for k, v in (s.get("params") or {}).items()},
                rationale=str(s.get("rationale", "")),
            )
        )
    return steps, None


class ProcedureSynthesizer:
    """NL → command sequence with simulator-validation retry loop."""

    def __init__(
        self,
        backend: LLMBackend,
        opcodes: list[str],
        param_values: list[str],
        validator: CommandValidator,
        max_validation_iters: int = 4,
        options: GenerateOptions | None = None,
    ) -> None:
        self.backend = backend
        self.opcodes = list(opcodes)
        self.param_values = list(param_values)
        self.validator = validator
        self.max_validation_iters = int(max_validation_iters)
        self.options = options or GenerateOptions(max_tokens=1024, temperature=0.1)

    def synthesize(self, request: str) -> ProcedureSynthResult:
        errors: list[str] = []
        steps: list[CommandStep] = []
        last_text = ""
        last_user = ""
        for it in range(self.max_validation_iters + 1):
            user_text = _user_prompt(request, self.opcodes, self.param_values, errors)
            last_user = user_text
            result = self.backend.generate(
                [SystemMessage(_SYNTH_SYSTEM), UserMessage(user_text)],
                options=self.options,
            )
            last_text = result.text
            parsed_steps, parse_err = _parse_steps(result.text)
            if parse_err is not None:
                errors = [f"parse error: {parse_err}"]
                continue
            # Vocab check
            unknown = [s.opcode for s in parsed_steps if s.opcode not in self.opcodes]
            if unknown:
                errors = [f"unknown opcodes: {unknown}"]
                continue
            # Simulator validator
            sim_errors = self.validator(parsed_steps)
            if sim_errors:
                errors = sim_errors
                steps = parsed_steps
                continue
            return ProcedureSynthResult(
                success=True,
                steps=parsed_steps,
                validation_errors=[],
                n_iterations=it + 1,
                final_prompt=user_text,
                raw_text=last_text,
            )
        return ProcedureSynthResult(
            success=False,
            steps=steps,
            validation_errors=errors,
            n_iterations=self.max_validation_iters + 1,
            final_prompt=last_user,
            raw_text=last_text,
        )
