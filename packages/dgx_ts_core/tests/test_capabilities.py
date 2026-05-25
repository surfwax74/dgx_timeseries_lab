"""Unit tests for Capabilities, FitMode, OutputKind."""

from __future__ import annotations

from dgx_ts_core.models import Capabilities, FitMode, OutputKind


def test_capabilities_construction() -> None:
    c = Capabilities(
        requires_pretraining=True,
        supports_streaming=False,
        supports_multivariate=True,
        native_context_len=1024,
        output_kind=OutputKind.PER_STEP,
        supports_peft=True,
        supports_export_onnx=True,
    )
    assert c.requires_pretraining
    assert c.output_kind is OutputKind.PER_STEP
    assert c.supports_peft


def test_fit_mode_values() -> None:
    assert FitMode.PRETRAIN.value == "pretrain"
    assert FitMode.FINETUNE.value == "finetune"
    assert FitMode.ZEROSHOT.value == "zeroshot"


def test_capabilities_is_hashable_and_frozen() -> None:
    a = Capabilities(
        requires_pretraining=False,
        supports_streaming=True,
        supports_multivariate=True,
        native_context_len=256,
        output_kind=OutputKind.PER_STEP,
    )
    assert {a}  # hashable → can live in a set
