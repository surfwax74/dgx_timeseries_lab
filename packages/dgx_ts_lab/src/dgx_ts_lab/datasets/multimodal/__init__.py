"""Phase 10 multi-modal datasets — telemetry + commands + event logs.

Three on-orbit operational modalities aligned to a common 1-second grid:

    telemetry    continuous channels (existing layered_synth shape)
    commands     discrete events bucketed to [count, last_opcode_id, last_param_id]
    logs         discrete events bucketed to [count, max_severity, last_source_id]

The MultiModalDataset emits TelemetryWindow whose tensor concatenates all
three modality groups along the channel dim. ``channel_modalities`` is a
parallel tuple telling the model which channels belong to which modality
(consumed by the SatMultiModalModule's three sub-encoders).
"""

from . import (  # noqa: F401  side-effect: register synth_multimodal_leo
    event_bucketer,
    synth_multimodal_leo,
)
from ._log_tokenizer import LogTokenizer, LogSeverity
from .event_bucketer import CommandEventBucketer, LogEventBucketer
from .multimodal_dataset import MultiModalDataset

__all__ = [
    "CommandEventBucketer",
    "LogEventBucketer",
    "LogSeverity",
    "LogTokenizer",
    "MultiModalDataset",
    "event_bucketer",
    "synth_multimodal_leo",
]
