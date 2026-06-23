"""FSDP strategy configuration for Lightning Fabric.

Reads knobs from ``TrainConfig.extra``:

    fsdp_auto_wrap_min_params:  int  (default 1e8) — wrap modules >= this size
    fsdp_activation_checkpointing: bool (default True)
    fsdp_sharding_strategy:     str  (default "FULL_SHARD")
    fsdp_cpu_offload:           bool (default False)
    fsdp_backward_prefetch:     str  (default "BACKWARD_PRE")

These map onto torch.distributed.fsdp.FullyShardedDataParallel options.
Fabric calls into FSDP under the hood when ``strategy: fsdp`` is set in
the trainer config; this module assembles the right keyword arguments.
"""

from __future__ import annotations

from typing import Any


def build_fsdp_strategy_kwargs(extra: dict[str, Any]) -> dict[str, Any]:
    """Return the strategy kwargs to pass to ``lightning.Fabric(strategy="fsdp", ...)``.

    On the dev box (CPU/single GPU) FSDP degrades to a no-op shard so the
    same config works everywhere — the only thing that matters at small
    scale is that the wrap policy and activation checkpointing fields are
    legal.
    """
    min_params = int(extra.get("fsdp_auto_wrap_min_params", 100_000_000))
    use_act_ckpt = bool(extra.get("fsdp_activation_checkpointing", True))
    sharding_strategy = str(extra.get("fsdp_sharding_strategy", "FULL_SHARD"))
    cpu_offload = bool(extra.get("fsdp_cpu_offload", False))
    backward_prefetch = str(extra.get("fsdp_backward_prefetch", "BACKWARD_PRE"))

    # Lazy import — torch FSDP types only exist when torch>=2.0 is installed.
    try:
        from functools import partial

        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        auto_wrap_policy = partial(
            size_based_auto_wrap_policy, min_num_params=min_params
        )
    except ImportError:
        auto_wrap_policy = None

    kwargs: dict[str, Any] = {
        "auto_wrap_policy": auto_wrap_policy,
        "sharding_strategy": sharding_strategy,
        "cpu_offload": cpu_offload,
        "backward_prefetch": backward_prefetch,
    }
    if use_act_ckpt:
        # Lightning Fabric understands activation_checkpointing_policy as a callable.
        # We probe for the private wrap helper to gate the assignment because older
        # torch versions don't ship it; the import itself is the availability check.
        try:
            from torch.distributed.fsdp.wrap import (  # noqa: F401
                _module_wrap_policy as _probe_wrap_policy,
            )
            kwargs["activation_checkpointing_policy"] = auto_wrap_policy
        except ImportError:
            pass

    # Strip None entries — Fabric is picky.
    return {k: v for k, v in kwargs.items() if v is not None}
