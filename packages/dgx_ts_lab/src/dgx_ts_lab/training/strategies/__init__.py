"""Distributed training strategies for the Fabric loop.

The strategy name in ``TrainConfig.strategy`` is what Lightning Fabric
consumes (e.g., ``"fsdp"``, ``"ddp"``, ``"deepspeed_stage_3"``). These
helper modules build the auxiliary configuration (auto-wrap policy,
activation checkpointing, etc.) that Fabric needs alongside.
"""

from .fsdp import build_fsdp_strategy_kwargs

__all__ = ["build_fsdp_strategy_kwargs"]
