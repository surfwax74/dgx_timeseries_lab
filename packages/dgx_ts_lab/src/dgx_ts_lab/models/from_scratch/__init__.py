"""From-scratch neural detectors.

Phase 2 deliverable: three transformer-based AD architectures, all
re-implemented from their papers (no vendored code) for air-gap cleanliness:

    - PatchTST + MAE             (`patchtst_mae.py`)
    - Anomaly Transformer        (`anomaly_transformer.py`)
    - DCdetector                 (`dcdetector.py`)

All three implement the neural-detector contract (see
`packages/dgx_ts_lab/src/dgx_ts_lab/training/fabric_loop.py`):

    detector.module: nn.Module
    detector.compute_loss(batch) -> Tensor
    detector.compute_score_batch(batch) -> Tensor  # (B, T) per-step scores

Importing this module registers all three with `DETECTOR_REGISTRY`.
"""

from . import (  # noqa: F401  side-effect: register all
    anomaly_transformer,
    dcdetector,
    patchtst_mae,
    sat_multimodal,
    sat_tsfm,
    sat_tsfm_multitask,
    sequence_transformer,
    subsystem_moe,
)
from .anomaly_transformer import AnomalyTransformerDetector
from .dcdetector import DCdetectorDetector
from .patchtst_mae import PatchTSTMAEDetector
from .sat_multimodal import SatMultiModalDetector
from .sat_tsfm import SatTSFMDetector
from .sat_tsfm_multitask import SatTSFMMultiTaskDetector
from .sequence_transformer import SequenceTransformerDetector
from .subsystem_moe import SubsystemMoEDetector

__all__ = [
    "AnomalyTransformerDetector",
    "DCdetectorDetector",
    "PatchTSTMAEDetector",
    "SatMultiModalDetector",
    "SatTSFMDetector",
    "SatTSFMMultiTaskDetector",
    "SequenceTransformerDetector",
    "SubsystemMoEDetector",
    "anomaly_transformer",
    "dcdetector",
    "patchtst_mae",
    "sat_multimodal",
    "sat_tsfm",
    "sat_tsfm_multitask",
    "sequence_transformer",
    "subsystem_moe",
]
