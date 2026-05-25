# configs/model/

Hydra group for detector selection.

## Files

| YAML | Registry key | Notes |
|---|---|---|
| `rolling_mean.yaml` | `rolling_mean` | Phase 0 baseline. No hyperparameters required (uses defaults). |
| `anomaly_transformer.yaml` *(Phase 2)* | `anomaly_transformer` | From-scratch transformer, association-discrepancy loss. |
| `dcdetector.yaml` *(Phase 2)* | `dcdetector` | Dual-attention contrastive. |
| `patchtst_mae.yaml` *(Phase 2)* | `patchtst_mae` | PatchTST backbone + masked reconstruction. |
| `chronos_small.yaml` *(Phase 3)* | `chronos` | Foundation model fine-tune. |
| `moirai_base.yaml` *(Phase 3)* | `moirai` | Multivariate foundation model. |

## Adding a new model config

1. Implement + register the detector (see [`packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md`](../../packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md)).
2. Add a YAML here with `_target_key: <your_key>` + hyperparameters.
3. Reference it: `dgx-ts train model=<your_yaml_name>`.
