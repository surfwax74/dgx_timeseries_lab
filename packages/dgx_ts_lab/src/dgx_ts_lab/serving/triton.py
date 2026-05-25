"""Triton Inference Server config emitter.

Writes a Triton model-store directory layout for the exported detector:

    <triton_store>/<model_name>/
        config.pbtxt                 raw-scores endpoint config
        1/model.onnx                 versioned ONNX (Triton convention)
    <triton_store>/<model_name>_with_threshold/    (optional)
        config.pbtxt
        1/model_with_threshold.onnx

For PINN-wrapped detectors (where physics + neural compose), this module
also emits a Triton ensemble pipeline:

    <triton_store>/<model_name>_ensemble/
        config.pbtxt                 ensemble: physics → residual_detector
"""

from __future__ import annotations

import shutil
from pathlib import Path


def write_triton_ensemble(
    *,
    model_name: str,
    onnx_paths: dict[str, Path],
    triton_store: Path,
    n_channels: int,
    window_length: int,
    max_batch_size: int = 16,
) -> dict[str, Path]:
    """Lay out a Triton model store from already-exported ONNX files.

    ``onnx_paths`` is the dict returned by ``serving.onnx_export.export_detector()``;
    keys are artifact names ("model", "model_with_threshold"), values are paths.

    Returns a dict of {triton_model_dirname: model_dir}.
    """
    triton_store = Path(triton_store)
    triton_store.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for artifact_name, src in onnx_paths.items():
        # Triton model name: "<model_name>" for the raw scores variant,
        # "<model_name>_with_threshold" for the baked one.
        if artifact_name == "model":
            triton_name = model_name
            out_name = "scores"
            out_dtype = "TYPE_FP32"
        elif artifact_name == "model_with_threshold":
            triton_name = f"{model_name}_with_threshold"
            out_name = "is_anomaly"
            out_dtype = "TYPE_BOOL"
        else:
            triton_name = f"{model_name}_{artifact_name}"
            out_name = artifact_name
            out_dtype = "TYPE_FP32"

        model_dir = triton_store / triton_name
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        target = version_dir / "model.onnx"
        shutil.copyfile(src, target)

        cfg = _build_config_pbtxt(
            name=triton_name,
            max_batch_size=max_batch_size,
            input_shape=[window_length, n_channels],
            output_name=out_name,
            output_shape=[window_length],
            output_dtype=out_dtype,
        )
        (model_dir / "config.pbtxt").write_text(cfg)
        written[triton_name] = model_dir
    return written


def _build_config_pbtxt(
    *,
    name: str,
    max_batch_size: int,
    input_shape: list[int],
    output_name: str,
    output_shape: list[int],
    output_dtype: str,
) -> str:
    """Render a minimal Triton config.pbtxt for an ONNX model."""
    return f"""name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "x"
    data_type: TYPE_FP32
    dims: {input_shape}
  }}
]

output [
  {{
    name: "{output_name}"
    data_type: {output_dtype}
    dims: {output_shape}
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""
