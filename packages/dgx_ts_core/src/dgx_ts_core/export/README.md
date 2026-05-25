# dgx_ts_core.export

**The MLOps lift-out contract.** When a trained detector is exported for downstream serving (`mm_mlops` or any other MLOps platform), exactly three artifacts come out:

1. `model.onnx` (or alternative format from `ExportFormat`) — the runtime artifact.
2. `model_card.yaml` — descriptive metadata: capabilities, intended subsystem, calibrated threshold, training metrics.
3. `feature_schema.yaml` — channel list, units, sample rate, normalization stats — **structurally identical to `TelemetryDataset.channels` + `DatasetStats`**, so training-time and serving-time feature assumptions stay in lockstep.

Downstream systems pip-install `dgx_ts_core` to deserialize these, and they need NOTHING else from this repo.

## Files

| File | What it defines |
|---|---|
| `model_card.py` | `ModelCard` dataclass. |
| `feature_schema.py` | `FeatureSchema` dataclass. |
| `formats.py` | `ExportFormat` enum: `ONNX`, `TORCHSCRIPT`, `TRITON`. |
| `__init__.py` | Re-exports. |

Concrete YAML emitters live in [`packages/dgx_ts_lab/src/dgx_ts_lab/serving/`](../../../../dgx_ts_lab/src/dgx_ts_lab/) (Phase 5).

## See also

- Parent: [`../README.md`](../README.md)
- Spec: [`docs/lift_to_mlops.md`](../../../../../docs/lift_to_mlops.md)
