"""LoRA / PEFT wrapping for neural detectors.

Phase 3 LoRA strategy (locked): apply LoRA to **attention Q/V only** for
T5-based models. The peft library handles the wrapping; this module is a
thin convenience wrapper that adapts our config knobs into peft's API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass
class LoraConfig:
    """Subset of peft.LoraConfig knobs we expose via Hydra."""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    # Default target-module names for T5-style models. Override for other
    # architectures (e.g., Llama uses q_proj/v_proj).
    target_modules: tuple[str, ...] = ("q", "v")
    bias: str = "none"          # "none" | "all" | "lora_only"


def wrap_with_lora(module: nn.Module, cfg: LoraConfig) -> nn.Module:
    """Wrap ``module`` with peft LoRA adapters using ``cfg``.

    Returns the same module reference (peft mutates in place) — caller
    should reassign so any wrappers around the module pick up the new
    parameter set.
    """
    # Lazy import keeps dgx_ts_lab importable on machines without peft installed.
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    peft_cfg = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias=cfg.bias,
    )
    return get_peft_model(module, peft_cfg)


def trainable_parameter_count(module: nn.Module) -> tuple[int, int]:
    """Returns ``(trainable, total)`` parameter counts. Useful for tests
    that verify LoRA only enables a small fraction of params."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total
