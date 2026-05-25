# Lift to MLOps

How a trained detector graduates from this experiment lab into the broader `mm_mlops` framework.

## The contract: three artifacts

`detector.export(format=ExportFormat.ONNX, path=...)` writes exactly three files into `path/`:

```
model.onnx            ← runtime artifact (or torchscript / triton config)
model_card.yaml       ← metrics, intended subsystem, calibrated threshold, capabilities
feature_schema.yaml   ← channel list, units, sample rate, normalization stats
```

That's it. `mm_mlops` consumes these three files and nothing else from this repo.

## Why this works for hot-swap

`feature_schema.yaml` has the **same structure** as `TelemetryDataset.channels + DatasetStats`. So:

- Training-time feature pipelines emit `channels` + `stats()`.
- Inference-time feature pipelines in `mm_mlops` consume `feature_schema.yaml`.
- They stay in lockstep automatically — schema drift is detectable by diff.

## What mm_mlops needs to do

Install only the lightweight interface package:

```
pip install dgx-ts-core
```

Then deserialize the artifacts:

```python
import yaml
from dgx_ts_core.export import ModelCard, FeatureSchema

with open("model_card.yaml") as f:
    card = ModelCard(**yaml.safe_load(f))
with open("feature_schema.yaml") as f:
    schema = FeatureSchema(**yaml.safe_load(f))
```

Load the ONNX model via `onnxruntime` (no torch, no lightning needed at serving time).

## Status

Phase 5 will implement the actual `export()` emitters and the integration test that loads the artifacts in a fresh venv with only `dgx-ts-core` installed.

Until then, the contract types (`ModelCard`, `FeatureSchema`, `ExportFormat`) are defined in `dgx_ts_core.export` and unit-tested.

## See also

- Interface defs: [`packages/dgx_ts_core/src/dgx_ts_core/export/README.md`](../packages/dgx_ts_core/src/dgx_ts_core/export/README.md)
- Air-gap distribution: [`air_gapped_setup.md`](air_gapped_setup.md) (artifacts move by sneakernet too)
