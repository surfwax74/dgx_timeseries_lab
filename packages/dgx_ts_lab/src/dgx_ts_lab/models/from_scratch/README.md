# dgx_ts_lab.models.from_scratch

From-scratch neural detectors — Phase 2 + Phase 4/6/8/10 extensions. All
re-implemented from the papers, no vendored code.

## Files

| File | Detector | Registry key | Notes |
|---|---|---|---|
| `patchtst_mae.py` | `PatchTSTMAEDetector` | `patchtst_mae` | Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023) + MAE adaptation for AD |
| `anomaly_transformer.py` | `AnomalyTransformerDetector` | `anomaly_transformer` | Xu et al., "Anomaly Transformer" (ICLR 2022) |
| `dcdetector.py` | `DCdetectorDetector` | `dcdetector` | Yang et al., "DCdetector" (KDD 2023) |
| `sat_tsfm.py` | `SatTSFMDetector` | `sat_tsfm` | Phase 4 — channel-flexible Sat-TSFM foundation model |
| `sat_tsfm_multitask.py` | `SatTSFMMultiTaskDetector` | `sat_tsfm_multitask` | Phase 6 — Sat-TSFM with fault / RUL / mode heads |
| `subsystem_moe.py` | `SubsystemMoEDetector` | `subsystem_moe` | Phase 4 — per-subsystem MoE gating |
| `sequence_transformer.py` | `SequenceTransformerDetector` | `sequence_transformer` | Phase 8 — BERT-style MLM over command sequences |
| `sat_multimodal.py` | `SatMultiModalDetector` | `sat_multimodal` | Phase 10 — three-stream (telem + cmd + log) cross-modal MAE |
| `_multimodal_blocks.py` | (internal) | — | `ModalityTypeEmbedding`, `PerModalitySelfAttn`, `SharedCrossModalStack` building blocks for `sat_multimodal` |

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

## Multi-modal input contract (`sat_multimodal`)

`SatMultiModalDetector` expects `batch["x"]` shaped `(B, T, C_total)` where
`C_total = n_telemetry_channels + 3 (commands) + 3 (logs)` in the order
emitted by `MultiModalDataset` (see [`datasets/multimodal/README.md`](../../datasets/multimodal/README.md)).
At `fit()` time it reads `n_telemetry_channels` from the dataset attribute
and remembers it; `compute_loss` / `compute_score_batch` split the batch
into the three modality streams accordingly. Mixing it with a non-multimodal
dataset is not supported.

## ONNX export status

Per `Capabilities.supports_export_onnx`:

- `patchtst_mae` ✓ — supports ONNX trace (Phase 5 implementation).
- `anomaly_transformer` ✗ — nested per-layer attention outputs complicate tracing; needs custom export.
- `dcdetector` ✗ — same complication.
- `sat_multimodal` ✗ — per-modality branching + dynamic mask tokens; pending.

## Where to look when…

| Need | Look at |
|---|---|
| Add a new neural detector | Pattern: subclass nothing, expose `module`/`compute_loss`/`compute_score_batch`, register with `DETECTOR_REGISTRY`. See [`docs/adding_a_model.md`](../../../../../../docs/adding_a_model.md). |
| Debug a training loss explosion | `compute_loss` in the specific file. The Fabric loop applies gradient clipping (default 1.0) — see `training/fabric_loop.py`. |
| Change architecture hyperparameters | The corresponding YAML in [`configs/model/`](../../../../../../configs/model/README.md). |
