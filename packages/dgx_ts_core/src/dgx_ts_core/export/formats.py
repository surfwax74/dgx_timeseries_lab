from __future__ import annotations

from enum import Enum


class ExportFormat(str, Enum):
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TRITON = "triton"
