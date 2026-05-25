# dgx_ts_lab.models.baseline

Classical (non-neural) detector baselines. Useful as comparison floors for the bake-off and as the simplest implementation of the `AnomalyDetector` Protocol.

## Files

| File | Detector | Registry key | Capabilities |
|---|---|---|---|
| `rolling_mean.py` | `RollingMeanDetector` | `rolling_mean` | per-step, streaming, no pretraining, no PEFT. Computes a causal rolling mean of size W and scores the residual against the calibrated residual-std from training. |

## How `rolling_mean` works

- **Fit**: walks training windows, computes the rolling-mean residual at every step, takes per-channel std of those residuals as the score normalizer.
- **Score**: `max_c |x[t,c] − rolling_mean[t,c]| / σ_residual_c`.

Designed to detect spike-like anomalies on top of smoothly-varying signals. Will NOT detect contextual or collective anomalies — that's what the Phase 2 neural detectors are for.

## See also

- Parent: [`../README.md`](../README.md) (full detector roadmap by phase)
