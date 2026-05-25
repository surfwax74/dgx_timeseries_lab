"""Generic ONNX exporter + per-detector wrapper dispatch.

Per the locked Phase 5 design, every detector that supports ONNX export
emits TWO artifacts:

    model.onnx                  takes (B, T, C) → returns (B, T) raw scores
    model_with_threshold.onnx   takes (B, T, C) → returns (B, T) is_anomaly bool
                                (only when capabilities.supports_export_threshold_baked)

Each detector class registers an `onnx_wrapper_factory(detector, threshold)`
function via the EXPORT_REGISTRY. The exporter looks up the right factory,
gets the wrapper nn.Module(s), and traces them through torch.onnx.export.

Opset 20 is the default (Phase 5 locked decision).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

EXPORT_REGISTRY: dict[type, Callable[..., dict[str, nn.Module]]] = {}


def register_onnx_wrapper(detector_cls: type):
    """Decorator: registers a factory that builds ONNX wrapper modules for
    the given detector class.

    The factory signature is:
        factory(detector, threshold: float | None) -> dict[str, nn.Module]
    The returned dict's keys are the artifact filenames (without .onnx):
        "model"                  → required, raw scores
        "model_with_threshold"   → optional, baked is_anomaly
    """

    def decorator(fn: Callable[..., dict[str, nn.Module]]) -> Callable[..., dict[str, nn.Module]]:
        EXPORT_REGISTRY[detector_cls] = fn
        return fn

    return decorator


def export_detector(
    detector: Any,
    *,
    output_dir: Path,
    threshold: float | None,
    n_channels: int,
    window_length: int,
    opset: int = 20,
    emit_threshold_variant: bool = True,
) -> dict[str, Path]:
    """Export a fitted detector to ONNX artifacts under ``output_dir``.

    Returns a dict of {artifact_name: written_path}.
    """
    factory = _lookup_factory(detector)
    wrappers = factory(detector, threshold)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dummy input shape; torch.onnx.export uses it only for tracing.
    dummy_x = torch.zeros(1, window_length, n_channels, dtype=torch.float32)

    written: dict[str, Path] = {}
    for name, wrapper in wrappers.items():
        if name == "model_with_threshold" and not emit_threshold_variant:
            continue
        if name == "model_with_threshold" and not detector.capabilities.supports_export_threshold_baked:
            continue
        target = output_dir / f"{name}.onnx"
        wrapper.eval()
        output_names = ["scores"] if name == "model" else ["is_anomaly"]
        # Use the legacy TorchScript-based exporter — battle-tested, no
        # onnxscript dependency, supports our dynamic-shape use case. The
        # newer dynamo-based exporter still has rough edges in torch 2.12.
        torch.onnx.export(
            wrapper,
            (dummy_x,),
            str(target),
            input_names=["x"],
            output_names=output_names,
            dynamic_axes={
                "x": {0: "batch", 1: "time"},
                output_names[0]: {0: "batch", 1: "time"},
            },
            opset_version=opset,
            dynamo=False,
        )
        written[name] = target
    return written


def _lookup_factory(detector) -> Callable[..., dict[str, nn.Module]]:
    cls = type(detector)
    if cls in EXPORT_REGISTRY:
        return EXPORT_REGISTRY[cls]
    # Walk the MRO so wrappers like PINNResidualDetector can fall back to their inner.
    for mro_cls in cls.__mro__:
        if mro_cls in EXPORT_REGISTRY:
            return EXPORT_REGISTRY[mro_cls]
    raise NotImplementedError(
        f"No ONNX wrapper factory registered for {cls.__name__}. "
        "Decorate a builder with @register_onnx_wrapper(<DetectorClass>) "
        "or set capabilities.supports_export_onnx=False on this detector."
    )


# ── Built-in wrapper factories ──────────────────────────────────────────
#
# Imported below so they self-register at module load. Order matters only
# for the MRO walk; each factory is independent.

from . import _wrappers  # noqa: E402, F401  side-effect: populate EXPORT_REGISTRY
