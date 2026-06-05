# dgx_ts_lab.models.behavior

Phase 8 behavior-based detectors. Distinguished from `from_scratch/` because these models score against per-entity baseline distributions rather than learning a single distribution.

## Files

| File | Detector | Registry key | Scoring |
|---|---|---|---|
| `operator_fingerprint.py` | `OperatorFingerprintDetector` | `operator_fingerprint` | Per-operator Mahalanobis distance. Train encoder + per-operator stats; at inference, distance from claimed operator's distribution = anomaly score. |

## How `operator_fingerprint` works

1. **Encoder**: small MLP that maps activity-window features `(F,)` → embedding `(D,)`.
2. **Pretraining**: contrastive loss — push same-operator embeddings together, push different-operator embeddings apart.
3. **Fit operator stats**: walk training data, per-operator compute embedding mean + covariance inverse, store as `nn.Module` buffers (`operator_means`, `operator_cov_inv`, `operator_valid`).
4. **Inference**: for each window, read claimed `operator_id` from `aux_labels`, compute embedding, return Mahalanobis distance to that operator's distribution.

Buffers are part of the module's state_dict, so save/load round-trips include the per-operator stats. Stats are also moved with the module to GPU via Fabric.

## Why Mahalanobis (vs cosine)

Locked Phase 8 decision. Mahalanobis captures per-operator variance — an operator with naturally noisy behavior tolerates more variation before being flagged. Fast at inference (one matmul per window), interpretable in standard-deviation units, ONNX-friendly.

Contrastive cosine + face-ID-style centroid is a possible Phase 8.5 upgrade if discrimination is insufficient.

## Two-step "fit"

Per the Lightning Fabric loop convention, `detector.fit()` only initializes the module. The Fabric loop then runs gradient updates. After training, we need a separate step to populate the per-operator stats — exposed as `OperatorFingerprintDetector.compute_operator_stats(dataset, length, stride)`.

For `dgx-ts train`, the trainer should call this after the Fabric loop completes when `requires_post_fit_stats=True` on the detector — current Phase 8 detector tests call it manually; auto-wiring lands as a small `LightningTrainer.fit()` extension.

## See also

- Cyber datasets: [`packages/dgx_ts_lab/src/dgx_ts_lab/datasets/cyber/README.md`](../../datasets/cyber/README.md)
- Phase 8 plan: [`docs/phase_plans/phases_6_through_11.md`](../../../../../../docs/phase_plans/phases_6_through_11.md)
