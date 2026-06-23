from __future__ import annotations

from enum import StrEnum


class ExportFormat(StrEnum):
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TRITON = "triton"
