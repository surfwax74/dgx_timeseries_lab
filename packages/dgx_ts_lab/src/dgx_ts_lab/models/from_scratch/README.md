# dgx_ts_lab.models.from_scratch

Phase 2 neural detectors — all re-implemented from the papers, no vendored code.

## Files

| File | Detector | Registry key | Paper |
|---|---|---|---|
| `patchtst_mae.py` | `PatchTSTMAEDetector` | `patchtst_mae` | Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023) + MAE adaptation for AD |
| `anomaly_transformer.py` | `AnomalyTransformerDetector` | `anomaly_transformer` | Xu et al., "Anomaly Transformer" (ICLR 2022) |
| `dcdetector.py` | `DCdetectorDetector` | `dcdetector` | Yang et al., "DCdetector" (KDD 2023) |

## The neural-detector contract

All three implement the contract documented in [`packages/dgx_ts_lab/src/dgx_ts_lab/training/README.md`](../../training/README.md):

```python
detector.module: torch.nn.Module                            # parameters live here
detector.compute_loss(batch: dict) -> torch.Tensor          # training step
detector.compute_score_batch(batch: dict) -> torch.Tensor   # (B, T) per-step scores
```

Their `Capabilities` declare `requires_pretraining=True`, so `LightningTrainer.fit()` routes them through the Fabric loop in [`fabric_loop.py`](../../training/fabric_loop.py).

## Normalization

Each module registers `norm_mean` and `norm_std` as **buffers** (not parameters). They're populated from `dataset.stats()` at `fit()` time. Buffers move to the right device automatically when Fabric calls `fabric.setup(module, optimizer)`, so there's no device-mismatch boilerplate.

## ONNX export status

Per `Capabilities.supports_export_onnx`:

- `patchtst_mae` ✓ — supports ONNX trace (Phase 5 implementation).
- `anomaly_transformer` ✗ — nested per-layer attention outputs complicate tracing; needs custom export.
- `dcdetector` ✗ — same complication.

## Where to look when…

| Need | Look at |
|---|---|
| Add a new neural detector | Pattern: subclass nothing, expose `module`/`compute_loss`/`compute_score_batch`, register with `DETECTOR_REGISTRY`. See [`docs/adding_a_model.md`](../../../../../../docs/adding_a_model.md). |
| Debug a training loss explosion | `compute_loss` in the specific file. The Fabric loop applies gradient clipping (default 1.0) — see `training/fabric_loop.py`. |
| Change architecture hyperparameters | The corresponding YAML in [`configs/model/`](../../../../../../configs/model/README.md). |
