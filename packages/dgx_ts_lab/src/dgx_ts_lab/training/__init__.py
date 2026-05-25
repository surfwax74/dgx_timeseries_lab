"""Training implementations. Importing this module registers all bundled
trainers with dgx_ts_core.registry.TRAINER_REGISTRY."""

from . import lightning_trainer  # noqa: F401  side-effect: register lightning
from .lightning_trainer import LightningTrainer

__all__ = ["LightningTrainer", "lightning_trainer"]
