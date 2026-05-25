# dgx_ts_lab.serving

Phase 5 — the MLOps lift-out boundary. Exports a trained detector to portable artifacts that downstream serving systems consume **with only `dgx_ts_core` + `onnxruntime` installed.**

## Files

| File | Role |
|---|---|
| `onnx_export.py` | `export_detector()` — torch.onnx.export with per-detector wrapper dispatch. Opset 20, dynamic batch+time axes. |
| `_wrappers.py` | Per-detector ONNX-traceable wrapper modules. Each detector class registers a factory via `@register_onnx_wrapper(MyDetector)`. |
| `model_card_writer.py` | `write_model_card()` — serializes `ModelCard` to YAML with provenance (git SHA, timestamp, platform, mlflow run id). |
| `feature_schema_writer.py` | `write_feature_schema()` — serializes `FeatureSchema` to YAML; mirrors `TelemetryDataset.channels + DatasetStats`. |
| `triton.py` | `write_triton_ensemble()` — lays out a Triton model-store directory with raw + threshold-baked endpoints. |

## The contract

`dgx-ts export model=... dataset=... +checkpoint=... +threshold=X +output_dir=DIR` writes:

```
DIR/
├── model.onnx                      raw per-step scores: (B, T, C) → (B, T) float32
├── model_with_threshold.onnx       (optional) is_anomaly: (B, T, C) → (B, T) bool
├── model_card.yaml                 detector metadata, metrics, threshold, capabilities
└── feature_schema.yaml             channel definitions + per-channel normalization
```

Add `+write_triton=true` to also write a Triton model-store layout under `+triton_store=...`.

## Per-detector ONNX support

| Detector | `supports_export_onnx` | `supports_export_threshold_baked` |
|---|---|---|
| `rolling_mean` | ✗ | ✗ |
| `patchtst_mae` | ✓ | ✓ |
| `anomaly_transformer` | ✗ *(nested attn outputs)* | ✗ |
| `dcdetector` | ✗ *(complex KL ops)* | ✗ |
| `chronos` | ✗ *(TODO: HF model.onnx_config)* | ✗ |
| `moment` | ✗ *(TODO)* | ✗ |
| `moirai` | ✗ | ✗ |
| `sat_tsfm` | ✓ | ✓ |
| `subsystem_moe` | ✗ *(per-channel routing complicates trace)* | ✗ |
| `pinn_residual` | ✗ *(use Triton ensemble — see PINN pattern)* | ✗ |

Flipping a `False` to `True` requires writing a wrapper factory in [`_wrappers.py`](_wrappers.py) and decorating with `@register_onnx_wrapper(<DetectorClass>)`. The factory returns a dict of `nn.Module` wrappers keyed by artifact name ("model", "model_with_threshold").

## Adding ONNX support for a new detector

1. Implement an ONNX-traceable `forward()` that takes one tensor (B, T, C) and returns one tensor (B, T). All ops must be in the supported opset.
2. Wrap in a tiny `nn.Module` for the raw-scores variant; another for the threshold-baked variant if `supports_export_threshold_baked=True`.
3. Decorate a factory: `@register_onnx_wrapper(MyDetector)` returning `{"model": raw, "model_with_threshold": baked}`.
4. Flip the detector's `Capabilities.supports_export_onnx=True` (and `supports_export_threshold_baked=True` if applicable).
5. Add a test exercising `export_detector()` → `onnxruntime.InferenceSession()` → `np.testing.assert_allclose` vs in-process detector.

## PINN ensemble pattern

PINN-wrapped detectors (`pinn_residual`) export as a Triton **ensemble**: the physics model and the inner neural detector are two separate Triton models composed via `config.pbtxt`. The ensemble endpoint takes raw `(B, T, C)` input and chains:

```
input → physics_subtract (Python backend) → residual → neural_detector (ONNX) → scores
```

This means `mm_mlops` consumers of PINN detectors need a Triton deployment, not just `onnxruntime`. See [`docs/serving_deployment.md`](../../../../../../docs/serving_deployment.md) for the full pipeline.

## See also

- Lift contract spec: [`docs/lift_to_mlops.md`](../../../../../../docs/lift_to_mlops.md)
- Consumer-side deployment guide: [`docs/serving_deployment.md`](../../../../../../docs/serving_deployment.md)
- Provisioning foundation models (Phase 3 prereq): [`docs/foundation_model_provisioning.md`](../../../../../../docs/foundation_model_provisioning.md)
