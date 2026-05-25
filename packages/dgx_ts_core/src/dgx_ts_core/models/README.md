# dgx_ts_core.models

Detector-side contracts: the `AnomalyDetector` Protocol that every model (classical baseline, from-scratch transformer, foundation-model adapter, physics residual wrapper) implements.

## Files

| File | What it defines |
|---|---|
| `detector.py` | `AnomalyDetector` Protocol + `FitResult` dataclass. |
| `capabilities.py` | `Capabilities` dataclass + `FitMode`, `OutputKind` enums. **The hot-swap mechanism: training and eval code branches on `capabilities`, never on `isinstance`.** |
| `scores.py` | `AnomalyScore` dataclass — detector output for one window. |
| `__init__.py` | Re-exports. |

## The hot-swap contract

Any new detector implements `AnomalyDetector` and declares a `Capabilities` object that describes what it can/can't do:

```python
Capabilities(
    requires_pretraining=True,        # needs a training loop?
    supports_streaming=False,
    supports_multivariate=True,
    native_context_len=4096,
    output_kind=OutputKind.PER_STEP,
    supports_peft=True,               # LoRA/adapter fine-tuning?
    supports_export_onnx=True,
)
```

Trainers (`dgx_ts_lab.training.LightningTrainer`) read these flags to decide which code path to run.

## See also

- Parent: [`../README.md`](../README.md)
- Implementations live in [`packages/dgx_ts_lab/src/dgx_ts_lab/models/`](../../../../dgx_ts_lab/src/dgx_ts_lab/models/README.md)
