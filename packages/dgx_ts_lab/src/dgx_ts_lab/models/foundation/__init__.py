"""Foundation-model adapters (Phase 3).

Wraps 2024-era pretrained time-series foundation models behind the
AnomalyDetector Protocol, supporting both zero-shot and LoRA fine-tuned
modes.

    chronos.py    Amazon Chronos (T5-based, loaded via transformers)
    moment.py     CMU MOMENT  (T5-encoder-based; full impl pending real weights)
    moirai.py     Salesforce Moirai (multivariate native; requires uni2ts package)
    timesfm.py    Google TimesFM 2.0 (decoder-only patched transformer, 500M)
    ttm.py        IBM Tiny Time Mixer r2 (MLP-Mixer, 1-5M, fast LoRA on 3080)
    time_moe.py   Maple728/TimeMoE-200M (MoE transformer, NeurIPS 2024)

All adapters share the per-channel-then-max scoring strategy from
`_base.py`: univariate foundation models forecast each channel
independently and the per-step score is the max channel-wise residual.

Importing this module registers all detectors with `DETECTOR_REGISTRY`.
"""

from . import (  # noqa: F401  side-effect: register every adapter
    chronos,
    moirai,
    moment,
    time_moe,
    timesfm,
    ttm,
)
from .chronos import ChronosDetector
from .moirai import MoiraiDetector
from .moment import MomentDetector
from .time_moe import TimeMoEDetector
from .timesfm import TimesFMDetector
from .ttm import TTMDetector

__all__ = [
    "ChronosDetector",
    "MoiraiDetector",
    "MomentDetector",
    "TTMDetector",
    "TimeMoEDetector",
    "TimesFMDetector",
    "chronos",
    "moirai",
    "moment",
    "time_moe",
    "timesfm",
    "ttm",
]
