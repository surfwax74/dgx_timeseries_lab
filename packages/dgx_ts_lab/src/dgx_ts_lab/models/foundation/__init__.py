"""Foundation-model adapters (Phase 3).

Wraps 2024-era pretrained time-series foundation models behind the
AnomalyDetector Protocol, supporting both zero-shot and LoRA fine-tuned
modes.

    chronos.py    Amazon Chronos (T5-based, loaded via transformers)
    moment.py     CMU MOMENT  (T5-encoder-based; full impl pending real weights)
    moirai.py     Salesforce Moirai (multivariate native; requires uni2ts package)

All three share the per-channel-then-max scoring strategy from `_base.py`:
univariate models forecast each channel independently and the per-step
score is the max channel-wise residual.

Importing this module registers all three with `DETECTOR_REGISTRY`.
"""

from . import (  # noqa: F401  side-effect: register chronos, moment, moirai
    chronos,
    moirai,
    moment,
)
from .chronos import ChronosDetector
from .moirai import MoiraiDetector
from .moment import MomentDetector

__all__ = [
    "ChronosDetector",
    "MoiraiDetector",
    "MomentDetector",
    "chronos",
    "moirai",
    "moment",
]
