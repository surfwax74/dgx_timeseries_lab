"""Recorded co-pilot Q&A session — produces a Markdown transcript for
the DGX procurement deck.

Runs a scripted set of operator questions against the chosen LLM backend
(default: vLLM Mixtral 8x22B on the DGX), captures each turn's tool
calls and final answer, and writes the result as a clean transcript
suitable for screenshots.

Usage:
    python scripts/dgx_showcase_copilot_qna.py \\
        --backend vllm \\
        --model mistralai/Mixtral-8x22B-Instruct-v0.1 \\
        --base-url http://localhost:8000/v1 \\
        --model-card runs/dgx_showcase/exports/sat_tsfm_xl/model_card.yaml \\
        --output runs/dgx_showcase/copilot_transcript.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from dgx_ts_lab.llm import build_backend
from dgx_ts_lab.llm.copilot import Copilot
from dgx_ts_lab.llm.rag import CosineRAGIndex, RAGDocument
from dgx_ts_lab.llm.telemetry_tools import CopilotContext, default_tool_registry


# ── Scripted operator scenario ──────────────────────────────────────────


_SCENARIO_QUESTIONS = [
    "We saw an anomaly fire on EPS bus voltage about 10 minutes ago. "
    "What channel had the strongest signal, and how does it compare to "
    "its typical range?",

    "Was there a commanding event in the same window? I'm worried about "
    "a possible operator error or an unauthorized command.",

    "Pull up the relevant procedure for an EPS bus undervoltage event — "
    "what's the recommended next action?",

    "Summarize this whole incident in one paragraph as if you were "
    "filing it in the daily ops log. Include the timestamp, root channel, "
    "command context, and recommended action.",
]


def _build_synthetic_context() -> CopilotContext:
    """Build a deterministic 'recent telemetry' context for the demo.

    In a real deployment this context comes from the live MLflow run +
    the trained model_card. For procurement-demo purposes we generate a
    representative slice so the LLM has something concrete to reason
    over even without a hot detector behind it.
    """
    rng = np.random.default_rng(0)
    T = 3600   # one hour @ 1 Hz
    channel_names = [
        "bus_v", "bus_i", "bat_soc", "solar_p", "tcs_temp", "pld_curr",
    ]
    tel = np.zeros((T, len(channel_names)), dtype=np.float32)
    # Baseline orbital sinusoids
    t = np.arange(T)
    tel[:, 0] = 28.0 + 0.6 * np.sin(2 * np.pi * t / 5400.0) + rng.normal(0, 0.02, T)
    tel[:, 1] = 1.2 + 0.1 * np.sin(2 * np.pi * t / 5400.0) + rng.normal(0, 0.01, T)
    tel[:, 2] = 0.85 + 0.05 * np.sin(2 * np.pi * t / 5400.0) + rng.normal(0, 0.005, T)
    tel[:, 3] = 80.0 + 50.0 * np.sin(2 * np.pi * t / 5400.0) + rng.normal(0, 1.5, T)
    tel[:, 4] = 18.0 + rng.normal(0, 0.15, T)
    tel[:, 5] = 0.8 + 0.3 * np.sin(2 * np.pi * t / 5400.0) + rng.normal(0, 0.02, T)
    # Inject an EPS bus undervoltage event 10 min ago (step 3000) lasting 30s
    tel[3000:3030, 0] -= 3.2
    tel[3000:3030, 1] += 0.8       # current spike — typical undervoltage signature
    # Anomaly scores from the (notional) trained detector
    scores = rng.uniform(0.0, 0.5, T).astype(np.float32)
    scores[3000:3030] = rng.uniform(4.5, 8.0, 30)
    # Tiny procedure corpus
    procs = [
        RAGDocument(
            "procedures/eps_undervoltage.md#0",
            "EPS Bus Undervoltage Response",
            "If bus voltage drops below 26.5V for >5s: (1) verify load-shed "
            "thresholds, (2) command MODE_SET to SAFE, (3) log incident in "
            "daily ops log, (4) page on-call EPS engineer if recovery fails "
            "within 60s.",
        ),
        RAGDocument(
            "procedures/safe_mode_entry.md#0",
            "Safe Mode Entry Checklist",
            "Pre-flight: confirm communications link. Issue: MODE_SET=SAFE "
            "with EPS_LOAD_SHED=1. Verify within 10s that bus voltage "
            "recovers to nominal 27.5-28.5V range.",
        ),
        RAGDocument(
            "procedures/operator_command_audit.md#0",
            "Operator Command Audit",
            "All operator commands within ±60s of an anomaly fire are "
            "automatically flagged for review. Use query_anomaly_history "
            "with a known anomaly step to correlate.",
        ),
    ]
    idx = CosineRAGIndex()
    idx.add_lexical(procs)
    return CopilotContext(
        telemetry=tel,
        channel_names=channel_names,
        sample_rate_hz=1.0,
        anomaly_scores=scores,
        anomaly_threshold=2.0,
        rag_index=idx,
    )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="mock",
                   choices=["anthropic", "vllm", "ollama", "llama_cpp", "mock"])
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model-card", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--max-tool-iters", type=int, default=4)
    return p


def _build_backend_from_args(args: argparse.Namespace):
    if args.backend == "mock":
        return build_backend("mock", echo=True)
    if args.backend == "anthropic":
        kwargs = {}
        if args.model:
            kwargs["model_id"] = args.model
        return build_backend("anthropic", **kwargs)
    if args.backend in ("vllm", "ollama"):
        if not args.model:
            raise SystemExit(f"{args.backend} requires --model")
        kwargs = {"model_id": args.model}
        if args.base_url:
            kwargs["base_url"] = args.base_url
        return build_backend(args.backend, **kwargs)
    if args.backend == "llama_cpp":
        if not args.model:
            raise SystemExit("llama_cpp requires --model (path to .gguf)")
        return build_backend("llama_cpp", model_path=args.model)
    raise SystemExit(f"unknown backend {args.backend}")


def main() -> int:
    args = _build_argparser().parse_args()
    backend = _build_backend_from_args(args)
    ctx = _build_synthetic_context()
    if args.model_card:
        ctx.model_card_path = Path(args.model_card)
    tools = default_tool_registry(context=ctx)
    copilot = Copilot(backend=backend, tools=tools,
                       max_tool_iters=args.max_tool_iters)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Co-Pilot Q&A Transcript")
    lines.append("")
    lines.append(f"- **Backend:** `{backend.name}`")
    lines.append(f"- **Model:**   `{backend.model_id}`")
    lines.append(f"- **Tools:**   {', '.join(t.name for t in tools.list_defs())}")
    lines.append(f"- **Recorded:** {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append("")

    total_t0 = time.time()
    for i, question in enumerate(_SCENARIO_QUESTIONS, 1):
        lines.append(f"---")
        lines.append(f"### Turn {i}")
        lines.append("")
        lines.append(f"**Operator:** {question}")
        lines.append("")
        t0 = time.time()
        try:
            turn = copilot.chat(question)
        except Exception as e:                                # noqa: BLE001
            lines.append(f"_ERROR: {type(e).__name__}: {e}_")
            lines.append("")
            continue
        elapsed = time.time() - t0
        if turn.tool_calls_made:
            lines.append(f"_Tool calls ({turn.n_tool_iterations} iter):_")
            for tc in turn.tool_calls_made:
                lines.append(f"  - `{tc['name']}({tc['args']})` → {tc['result_chars']} chars")
            lines.append("")
        lines.append(f"**Co-pilot:** {turn.text}")
        lines.append("")
        lines.append(f"_response: {elapsed:.2f}s_")
        lines.append("")
        print(f"  Q{i} done in {elapsed:.1f}s "
              f"({turn.n_tool_iterations} tool iter, "
              f"{len(turn.tool_calls_made)} calls)")

    total = time.time() - total_t0
    lines.append("---")
    lines.append(f"**Total session: {total:.2f}s across {len(_SCENARIO_QUESTIONS)} turns**")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
