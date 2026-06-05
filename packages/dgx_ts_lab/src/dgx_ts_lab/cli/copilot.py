"""dgx-ts copilot — interactive LLM ops co-pilot REPL (Phase 11).

Usage:
    dgx-ts copilot --backend {anthropic,vllm,ollama,llama_cpp,mock}
                   [--model MODEL_ID]
                   [--procedures DIR]
                   [--model-card PATH]

The REPL keeps multi-turn chat state, calls tools when the LLM requests
them, and prints both the final assistant message and a per-turn audit
trail. Type ``/reset`` to drop history, ``/exit`` (or Ctrl-D) to quit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..llm import build_backend
from ..llm.copilot import Copilot
from ..llm.rag import CosineRAGIndex, load_procedures_directory
from ..llm.telemetry_tools import CopilotContext, default_tool_registry


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dgx-ts copilot",
        description="LLM ops co-pilot REPL (Phase 11).",
    )
    p.add_argument(
        "--backend",
        default="mock",
        choices=["anthropic", "vllm", "ollama", "llama_cpp", "mock"],
    )
    p.add_argument("--model", default=None, help="Model ID / path for the chosen backend")
    p.add_argument("--base-url", default=None, help="HTTP base URL (vllm/ollama)")
    p.add_argument("--procedures", default=None, help="Directory of *.md procedures for RAG")
    p.add_argument("--model-card", default=None, help="Path to a Phase 5 model_card.yaml")
    p.add_argument("--system", default=None, help="Override system prompt")
    return p


def _build_backend_from_args(args: argparse.Namespace):
    if args.backend == "anthropic":
        kwargs = {}
        if args.model:
            kwargs["model_id"] = args.model
        return build_backend("anthropic", **kwargs)
    if args.backend == "vllm":
        if not args.model:
            raise SystemExit("vllm backend requires --model")
        kwargs = {"model_id": args.model}
        if args.base_url:
            kwargs["base_url"] = args.base_url
        return build_backend("vllm", **kwargs)
    if args.backend == "ollama":
        if not args.model:
            raise SystemExit("ollama backend requires --model")
        kwargs = {"model_id": args.model}
        if args.base_url:
            kwargs["base_url"] = args.base_url
        return build_backend("ollama", **kwargs)
    if args.backend == "llama_cpp":
        if not args.model:
            raise SystemExit("llama_cpp backend requires --model (path to .gguf)")
        return build_backend("llama_cpp", model_path=args.model)
    return build_backend("mock")


def _build_context_from_args(args: argparse.Namespace) -> CopilotContext:
    ctx = CopilotContext()
    if args.procedures:
        idx = CosineRAGIndex()
        docs = load_procedures_directory(args.procedures)
        if docs:
            idx.add_lexical(docs)
        ctx.rag_index = idx
    if args.model_card:
        ctx.model_card_path = Path(args.model_card)
    return ctx


def run() -> None:
    args = _build_argparser().parse_args()
    backend = _build_backend_from_args(args)
    tools = default_tool_registry(context=_build_context_from_args(args))
    copilot = Copilot(backend=backend, tools=tools, system_prompt=args.system)

    print(f"dgx-ts copilot — backend={backend.name} model={backend.model_id}")
    print("Type /reset to clear history, /exit (or Ctrl-D) to quit.")
    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_text:
            continue
        if user_text in ("/exit", "/quit"):
            return
        if user_text == "/reset":
            copilot.reset()
            print("[history reset]")
            continue
        try:
            turn = copilot.chat(user_text)
        except Exception as e:                                  # noqa: BLE001
            print(f"[error: {type(e).__name__}: {e}]", file=sys.stderr)
            continue
        if turn.tool_calls_made:
            print(f"[{turn.n_tool_iterations} tool iter(s); "
                  f"{len(turn.tool_calls_made)} tool call(s)]")
        print(turn.text)


if __name__ == "__main__":
    run()
