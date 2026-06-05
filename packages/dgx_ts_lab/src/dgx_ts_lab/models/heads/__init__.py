"""Phase 6 multi-task heads.

A ``TaskHead`` is a small ``nn.Module`` that attaches to a shared encoder's
per-step embedding ``(B, T, D)`` and produces a task-specific output. Each
head defines its own loss + metrics; the multi-task wrapper sums weighted
losses across active heads.

Importing this module registers the bundled heads with
``dgx_ts_core.registry.HEAD_REGISTRY``.
"""

from . import fault_classifier, mode_predictor, rul_regressor  # noqa: F401  registrations
from ._base import TaskHead
from .fault_classifier import FaultClassifierHead
from .mode_predictor import ModePredictorHead
from .rul_regressor import RULRegressorHead

__all__ = [
    "FaultClassifierHead",
    "ModePredictorHead",
    "RULRegressorHead",
    "TaskHead",
    "fault_classifier",
    "mode_predictor",
    "rul_regressor",
]
