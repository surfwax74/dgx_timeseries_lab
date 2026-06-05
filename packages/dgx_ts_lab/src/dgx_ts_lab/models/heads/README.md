# dgx_ts_lab.models.heads

Phase 6 — task heads that attach to a shared encoder (e.g., Sat-TSFM) to enable multi-task learning beyond anomaly detection.

## Files

| File | Head | Registry key | Task |
|---|---|---|---|
| `_base.py` | `TaskHead` | — | Base class — `compute_loss`, `compute_metrics`, `forward` |
| `fault_classifier.py` | `FaultClassifierHead` | `fault_classifier` | Per-step categorical: which fault type is active (or none)? |
| `rul_regressor.py` | `RULRegressorHead` | `rul_regressor` | Per-step regression: seconds-to-next-fault (log1p, masked MSE) |
| `mode_predictor.py` | `ModePredictorHead` | `mode_predictor` | Per-step categorical: what mode will the spacecraft be in 60 s from now? |

## How heads attach to an encoder

Heads consume **per-step pooled embeddings** `(B, T, D)` produced by the encoder. For Sat-TSFM the relevant method is `SatTSFMModule.encode_pooled_steps(x)`:

1. Encode with the full joint-attention transformer → `(B, C*N, D)`.
2. Reshape to `(B, C, N, D)` and mean-pool across channels → `(B, N, D)`.
3. Broadcast each patch embedding to its `patch_len` constituent timesteps → `(B, T_trunc, D)`.

Each head's `forward` takes that per-step tensor and produces task output (logits, regression, etc.).

## Where labels come from

Multi-task labels live under `TelemetryWindow.aux_labels` as a dict keyed by task name:

```python
aux_labels = {
    "fault_type": np.array([0, 0, 3, 3, 3, 0, ...], dtype=int64),   # 0 == no fault, 3 == dropout
    "rul":        np.array([300.0, 299.0, ..., 1e9, ...], dtype=float32),
    "next_mode":  np.array([0, 0, 1, 1, ..., -1, -1], dtype=int64),  # -1 == out of horizon
}
```

The label generator lives at [`datasets/synthetic/layered/labels.py`](../../datasets/synthetic/layered/labels.py) — it derives all three arrays from the dataset's `fault_log` + `mode_trace`. Enable via `LayeredSyntheticDataset(..., emit_multitask_labels=True)`.

## Adding a new head

1. Subclass `TaskHead` in a new file.
2. Set class-level `name` (used in metrics) and `label_key` (where to read targets from `batch["aux_labels"]`).
3. Implement `forward`, `compute_loss`, `compute_metrics`.
4. Register with `@HEAD_REGISTRY.register("my_head_key")`.
5. Add `from . import my_head` to this `__init__.py`.
6. Add a config under `configs/model/heads/my_head_key.yaml`.

## Multi-task wrapper usage

```yaml
# configs/model/sat_tsfm_multitask.yaml
_target_key: sat_tsfm_multitask
window_length: 256
d_model: 128
# ... other Sat-TSFM backbone params
heads:
  - {key: fault_classifier, num_classes: 8}
  - {key: rul_regressor}
  - {key: mode_predictor, num_modes: 6}
ad_loss_weight: 1.0
```

Joint loss = `ad_loss_weight * AD_recon_MSE + Σ head.loss_weight * head.compute_loss()`. Per-task metrics surface in `FitResult.metadata["per_task_metrics"]`.

## See also

- Wrapper: [`packages/dgx_ts_lab/src/dgx_ts_lab/models/from_scratch/sat_tsfm_multitask.py`](../from_scratch/sat_tsfm_multitask.py)
- Label generator: [`packages/dgx_ts_lab/src/dgx_ts_lab/datasets/synthetic/layered/labels.py`](../../datasets/synthetic/layered/labels.py)
- Phase 6 plan: [`docs/phase_plans/phases_6_through_11.md`](../../../../../../docs/phase_plans/phases_6_through_11.md)
