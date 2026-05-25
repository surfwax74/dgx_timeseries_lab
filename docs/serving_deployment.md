# Serving deployment

How a `mm_mlops` consumer (or any other serving platform) takes a `dgx-ts export` artifact and puts it into production.

## The contract

Every export produces:

```
exported/<detector_name>/
├── model.onnx                      raw scores
├── model_with_threshold.onnx       optional, is_anomaly bool
├── model_card.yaml                 metadata + provenance
└── feature_schema.yaml             channel layout + normalization
```

(plus an optional `triton_models/` directory if `+write_triton=true`).

## Consumer-side dependencies

The consumer needs ONLY:

```
dgx-ts-core         # interfaces + dataclasses, no torch
onnxruntime         # CPU or GPU EP
numpy
pyyaml
```

No torch, no lightning, no transformers, no peft. **`dgx-ts-core` ships from the same repo via `pip install dgx-ts-core`** (or `uv add dgx-ts-core` in the consumer's project).

## Minimal consumer code

```python
import numpy as np
import onnxruntime as ort
import yaml
from dgx_ts_core.export import FeatureSchema, ModelCard
from dgx_ts_core.models import Capabilities, OutputKind

ART = "exported/patchtst_mae"

# 1. Load metadata
card = yaml.safe_load(open(f"{ART}/model_card.yaml"))
schema = yaml.safe_load(open(f"{ART}/feature_schema.yaml"))

# 2. Sanity-check the feature pipeline against the schema
expected_channels = [c["name"] for c in schema["channels"]]
# (your feature pipeline must produce these channels, in this order, at
# the declared sample rate)
assert your_feature_pipeline.channels == expected_channels

# 3. Load the model
sess = ort.InferenceSession(f"{ART}/model.onnx")

# 4. Inference on a window
x = np.asarray(your_window, dtype=np.float32)   # shape (B, T, C)
scores = sess.run(["scores"], {"x": x})[0]      # shape (B, T) float32

# 5. Apply threshold (or use the baked variant)
threshold = card["calibrated_threshold"]
is_anomaly = scores > threshold
```

## Pick raw vs baked

| Variant | Output | When to use |
|---|---|---|
| `model.onnx` (raw) | `scores: (B, T) float32` | You want to retune the threshold post-deploy (e.g., adapt to operational SLAs) without re-exporting. |
| `model_with_threshold.onnx` (baked) | `is_anomaly: (B, T) bool` | Zero-config serving; threshold is fixed at export time. Re-tuning requires re-export. |

Most production setups ship both and pick at request time.

## Triton deployment

When `dgx-ts export +write_triton=true` is used, you get a model-store layout:

```
triton_models/
├── patchtst_mae/
│   ├── config.pbtxt
│   └── 1/model.onnx
└── patchtst_mae_with_threshold/
    ├── config.pbtxt
    └── 1/model_with_threshold.onnx
```

Deploy:

```bash
tritonserver \
    --model-store=/path/to/triton_models \
    --model-control-mode=explicit \
    --load-model=patchtst_mae \
    --load-model=patchtst_mae_with_threshold
```

Call via gRPC/HTTP:

```python
import tritonclient.http as httpclient
client = httpclient.InferenceServerClient(url="localhost:8000")
input0 = httpclient.InferInput("x", x.shape, "FP32")
input0.set_data_from_numpy(x)
result = client.infer("patchtst_mae", [input0])
scores = result.as_numpy("scores")
```

## PINN-wrapped detectors

PINN detectors (`pinn_residual`) ship as a Triton **ensemble**:

```
triton_models/
├── orbital_physics/        Python-backend model: subtracts physics from input
│   └── config.pbtxt
├── patchtst_mae/           neural residual detector (ONNX)
│   ├── config.pbtxt
│   └── 1/model.onnx
└── orbital_pinn_patchtst_ensemble/   composed pipeline
    └── config.pbtxt
```

Consumers call the ensemble endpoint with raw `(B, T, C)` input; Triton routes it through the physics step then the neural step. The serving platform needs Triton — not just onnxruntime.

*(Phase 5 ships the file layout for non-PINN detectors today. The PINN ensemble Python-backend physics model lands in a follow-up; the trial expands `serving/triton.py` with an ensemble emitter.)*

## Versioning

`dgx-ts export model=models:/sat_tsfm/Production` always exports the Production stage from the MLflow Registry. To export a specific version:

```bash
dgx-ts export model=models:/sat_tsfm/3 ...    # version 3 explicitly
```

The model card records the version in `card.detector_version` and `card.extra.mlflow_run_id` (when available).

## Compatibility matrix

| Consumer runtime | Raw scores ONNX | Threshold-baked ONNX | Triton ensemble (PINN) |
|---|---|---|---|
| onnxruntime CPU | ✓ | ✓ | ✗ (needs Triton) |
| onnxruntime CUDA | ✓ | ✓ | ✗ |
| onnxruntime TensorRT EP | ✓ | ✓ | ✗ |
| Triton (any backend) | ✓ | ✓ | ✓ |
| Browser (onnx-web) | ✓ | ✓ | ✗ |
| Edge (onnxruntime-mobile) | ✓ (small models only) | ✓ | ✗ |

## See also

- Lift contract spec: [`lift_to_mlops.md`](lift_to_mlops.md)
- Consumer-side artifact reference: [`packages/dgx_ts_core/src/dgx_ts_core/export/README.md`](../packages/dgx_ts_core/src/dgx_ts_core/export/README.md)
- ONNX export details: [`packages/dgx_ts_lab/src/dgx_ts_lab/serving/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/serving/README.md)
