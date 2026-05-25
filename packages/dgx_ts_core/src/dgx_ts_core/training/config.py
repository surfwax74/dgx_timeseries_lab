from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    """Generic training config consumed by Trainer implementations.

    Concrete Trainer implementations may subclass to add framework-specific
    fields. Hydra YAMLs in dgx_ts_lab map onto this shape.
    """

    max_epochs: int = 10
    batch_size: int = 64
    window_length: int = 256
    window_stride: int = 128
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = "auto"              # auto | cpu | cuda
    precision: str = "32-true"        # passed to Lightning Fabric
    strategy: str = "auto"            # auto | ddp | fsdp | deepspeed_*
    num_workers: int = 4
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    extra: dict[str, Any] = field(default_factory=dict)
