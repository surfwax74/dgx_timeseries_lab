"""Phase 5 serving + export.

The MLOps lift-out boundary. Each detector's ``export()`` method writes
three (or four) artifacts that downstream serving systems consume with
ONLY ``dgx_ts_core`` + ``onnxruntime`` + ``numpy`` + ``pyyaml`` installed:

    model.onnx                  raw-score ONNX model
    model_with_threshold.onnx   (optional) threshold-baked is_anomaly variant
    model_card.yaml             metrics, threshold, capabilities, provenance
    feature_schema.yaml         channels, units, sample rate, normalization

Plus an optional Triton model-store layout for direct deployment.
"""

from .feature_schema_writer import write_feature_schema
from .model_card_writer import write_model_card
from .onnx_export import (
    EXPORT_REGISTRY,
    export_detector,
    register_onnx_wrapper,
)
from .triton import write_triton_ensemble

__all__ = [
    "EXPORT_REGISTRY",
    "export_detector",
    "register_onnx_wrapper",
    "write_feature_schema",
    "write_model_card",
    "write_triton_ensemble",
]
