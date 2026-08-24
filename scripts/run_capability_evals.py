"""Run the capability-escalation suite across one or more LLM backends.

Answers the procurement question with evidence: **at what point does a
small local model stop being good enough?**

Typical uses
------------

Smoke it with no GPUs and no network (mock backend, CI-safe):

    python scripts/run_capability_evals.py --backend mock

Compare a small local model against the DGX-hosted frontier model.
Serve each in turn (or on different ports) and point at both:

    python scripts/run_capability_evals.py \\
        --model ollama:gemma4:e4b \\
        --model vllm:google/gemma-4-26B-A4B-it@http://localhost:8000/v1 \\
        --model vllm:meta-llama/Llama-3.1-405B-Instruct@http://localhost:8001/v1 \\
        --output runs/capability_evals

Run only the tiers that discriminate (skip the tier-1 control):

    python scripts/run_capability_evals.py --model ... --tiers 3,4

Add your own team scenarios alongside the shipped ones:

    python scripts/run_capability_evals.py --scenarios configs/llm_evals \\
        --scenarios configs/llm_evals_team

Output
------
``<output>/capability_report.md``   — escalation table, pass matrix, per-scenario detail
``<output>/capability_report.json`` — same data, machine-readable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_model_spec(spec: str):
    """Parse ``kind:model_id[@base_url]`` into a backend instance.

    Examples
    --------
    ``mock``
    ``ollama:gemma4:e4b``
    ``vllm:google/gemma-4-26B-A4B-it@http://localhost:8000/v1``
    ``llama_cpp:data/llm_weights/gemma-4-E2B-it.Q4_K_M.gguf``
    ``anthropic:claude-sonnet-4-5``
    """
    from dgx_ts_lab.llm import build_backend

    base_url = None
    if "@" in spec:
        spec, base_url = spec.rsplit("@", 1)

    if ":" in spec:
        kind, model_id = spec.split(":", 1)
    else:
        kind, model_id = spec, None

    kind_norm = kind.lower().replace("-", "_")
    kwargs: dict = {}
    if model_id:
        # llama.cpp is pointed at a GGUF path, not a hub model id; the mock
        # backend takes neither.
        if kind_norm in ("llama_cpp", "llamacpp"):
            kwargs["model_path"] = model_id
        elif kind_norm != "mock":
            kwargs["model_id"] = model_id
    if base_url and kind_norm in ("vllm", "ollama"):
        kwargs["base_url"] = base_url
    return build_backend(kind, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", action="append", default=[], metavar="SPEC",
        help="Backend spec kind:model_id[@base_url]. Repeat for a comparison matrix.",
    )
    parser.add_argument(
        "--backend", action="append", default=[], metavar="KIND",
        help="Shorthand for a backend with its config default (e.g. --backend mock).",
    )
    parser.add_argument(
        "--scenarios", action="append", default=[], type=Path, metavar="DIR",
        help="Scenario directory. Repeat to combine shipped + team scenarios. "
             "Default: configs/llm_evals",
    )
    parser.add_argument(
        "--procedures", type=Path, default=REPO_ROOT / "docs" / "procedures",
        help="Directory of procedure markdown to build the RAG index from.",
    )
    parser.add_argument(
        "--tiers", type=str, default="",
        help="Comma-separated tiers to run, e.g. '3,4'. Default: all.",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "runs" / "capability_evals",
        help="Where to write capability_report.{md,json}.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List the scenarios that would run, then exit without calling any model.",
    )
    args = parser.parse_args()

    from dgx_ts_lab.llm.evals import load_scenario_dir, run_matrix, write_reports

    # ── Scenarios ────────────────────────────────────────────────────
    scenario_dirs = args.scenarios or [REPO_ROOT / "configs" / "llm_evals"]
    tiers = [int(t) for t in args.tiers.split(",") if t.strip()] if args.tiers else None

    scenarios = []
    seen_ids: set[str] = set()
    for d in scenario_dirs:
        for s in load_scenario_dir(d, tiers=tiers):
            if s.id in seen_ids:
                print(f"WARNING: duplicate scenario id {s.id!r} in {d} — skipping", file=sys.stderr)
                continue
            seen_ids.add(s.id)
            scenarios.append(s)
    scenarios.sort(key=lambda s: (s.tier, s.id))

    if not scenarios:
        print("No scenarios matched. Check --scenarios / --tiers.", file=sys.stderr)
        return 2

    if args.list:
        print(f"{len(scenarios)} scenario(s) from {[str(d) for d in scenario_dirs]}:\n")
        for s in scenarios:
            print(f"  T{s.tier}  {s.id:<34} {s.title}")
            if s.expected_min_tier:
                print(f"        expected to need: {s.expected_min_tier}")
        return 0

    # ── Backends ─────────────────────────────────────────────────────
    specs = list(args.model) + list(args.backend)
    if not specs:
        print("No --model/--backend given; defaulting to the mock backend.", file=sys.stderr)
        specs = ["mock"]

    backends = []
    for spec in specs:
        try:
            backends.append(_parse_model_spec(spec))
        except Exception as e:                                # noqa: BLE001
            print(f"ERROR: could not build backend {spec!r}: {e}", file=sys.stderr)
            return 3

    if not args.procedures.exists():
        print(f"ERROR: procedures dir not found: {args.procedures}", file=sys.stderr)
        return 4

    print(
        f"Running {len(scenarios)} scenario(s) x {len(backends)} backend(s) "
        f"= {len(scenarios) * len(backends)} cells\n"
        f"  procedures: {args.procedures}\n"
    )

    runs = run_matrix(
        scenarios=scenarios,
        backends=backends,
        procedures_dir=args.procedures,
    )

    paths = write_reports(runs, scenarios, args.output)

    # ── Console summary ──────────────────────────────────────────────
    from dgx_ts_lab.llm.evals import TIER_NAMES, escalation_point

    print("\n" + "=" * 62)
    print(" Escalation points")
    print("=" * 62)
    models = sorted({r.model_id or r.backend_name for r in runs})
    for m in models:
        tier, why = escalation_point(runs, m)
        label = f"Tier {tier} ({TIER_NAMES.get(tier, '?')})" if tier else "none found"
        print(f"  {m:<44} {label}")
        print(f"      {why}")
    print()
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON:     {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
