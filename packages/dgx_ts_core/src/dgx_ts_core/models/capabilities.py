from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FitMode(str, Enum):
    PRETRAIN = "pretrain"
    """From-scratch self-supervised training on unlabeled telemetry."""

    FINETUNE = "finetune"
    """Adapt a pretrained foundation model — full or PEFT (LoRA, adapters)."""

    ZEROSHOT = "zeroshot"
    """No parameter updates; only a threshold is calibrated on val data."""


class OutputKind(str, Enum):
    PER_STEP = "per_step"
    """Detector emits one anomaly score per timestep — required for
    sub-window localization."""

    PER_WINDOW = "per_window"
    """Detector emits one anomaly score per window."""


@dataclass(frozen=True, slots=True)
class Capabilities:
    """A detector's self-declared capabilities.

    Trainers and evaluators branch on these fields rather than on isinstance()
    checks of the concrete class. This is the mechanism that lets a Chronos
    foundation model and a from-scratch Anomaly Transformer plug into the
    same training loop without special-casing.
    """

    requires_pretraining: bool
    supports_streaming: bool
    supports_multivariate: bool
    native_context_len: int
    output_kind: OutputKind
    supports_peft: bool = False
    supports_export_onnx: bool = False
    supports_zero_shot: bool = False
    # Phase 5: can the detector emit a threshold-baked ONNX variant
    # that outputs is_anomaly:bool directly instead of raw scores?
    supports_export_threshold_baked: bool = False
