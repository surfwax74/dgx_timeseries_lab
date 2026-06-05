"""Backend factory — pick an LLMBackend from a config dict.

The Hydra configs under `configs/llm/*.yaml` shape match `**kwargs` for
the corresponding backend's constructor; this factory just dispatches.
Keeps per-backend SDK imports lazy: only the chosen backend's module is
touched.
"""

from __future__ import annotations

from typing import Any

from .backend import LLMBackend
from ._mock_backend import MockBackend


def build_backend(kind: str, **kwargs: Any) -> LLMBackend:
    """Build an LLMBackend by name.

    Recognized kinds: anthropic | vllm | ollama | llama_cpp | mock.
    """
    k = kind.lower().replace("-", "_")
    if k == "anthropic":
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend(**kwargs)
    if k == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(**kwargs)
    if k == "ollama":
        from .ollama_backend import OllamaBackend
        return OllamaBackend(**kwargs)
    if k in ("llama_cpp", "llamacpp"):
        from .llama_cpp_backend import LlamaCppBackend
        return LlamaCppBackend(**kwargs)
    if k == "mock":
        return MockBackend(**kwargs)
    raise ValueError(
        f"unknown LLM backend kind: {kind!r}. "
        f"Expected one of: anthropic, vllm, ollama, llama_cpp, mock."
    )
