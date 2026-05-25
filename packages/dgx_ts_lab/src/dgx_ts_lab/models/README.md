# dgx_ts_lab.models

Concrete `AnomalyDetector` implementations. Importing this package self-registers every detector with `dgx_ts_core.registry.DETECTOR_REGISTRY`.

## Subdirs and phase status

| Subdir | What lives here | Phase |
|---|---|---|
| [`baseline/`](baseline/README.md) | Classical baselines (rolling mean residual today). | 0 ✓ |
| `from_scratch/` | Anomaly Transformer, DCdetector, PatchTST+MAE, TranAD. | 2 (planned) |
| `foundation/` | Chronos, MOMENT, Moirai, TimesFM adapters (zero-shot + LoRA). | 3 (planned) |
| `physics/` | PINN residual wrappers, NVIDIA Modulus integration. | 4 (planned) |

## Adding a new detector

1. Implement the `AnomalyDetector` Protocol from `dgx_ts_core.models`.
2. Declare honest `Capabilities` — `requires_pretraining`, `supports_streaming`, `output_kind`, `native_context_len`, `supports_peft`, `supports_export_onnx`. Trainers branch on these.
3. Decorate a factory with `@DETECTOR_REGISTRY.register("my_key")`.
4. Add `from . import my_subpkg` (or `from . import my_module`) to this `__init__.py` so import-time self-registration triggers.
5. Add `configs/model/my_key.yaml` with `_target_key: my_key` plus any hyperparameters.
6. Add tests.

See [`docs/adding_a_model.md`](../../../../../docs/adding_a_model.md) for the full walkthrough.
