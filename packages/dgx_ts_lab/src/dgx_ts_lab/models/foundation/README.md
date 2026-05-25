# dgx_ts_lab.models.foundation

Phase 3 — pretrained foundation-model adapters.

## Files

| File | Role |
|---|---|
| `_loader.py` | Dual-path weight resolver: MLflow Registry primary, `data/models/` fallback. |
| `_base.py` | `ForecastingDetector` — shared sliding-window inference, per-channel-then-max scoring, normalization buffers. |
| `chronos.py` | Amazon Chronos (T5-based). **Full working impl** via HF `transformers`. Registry key `chronos`. |
| `moment.py` | CMU MOMENT (T5-encoder + recon head). **Working impl** via HF `transformers`; closer to paper once real weights are loaded. Registry key `moment`. |
| `moirai.py` | Salesforce Moirai (multivariate-native). **Architectural shell**; needs `uni2ts` package + weights to do real forecasting. Errors at fit() if package missing. Registry key `moirai`. |

All three implement the neural-detector contract from [`packages/dgx_ts_lab/src/dgx_ts_lab/training/README.md`](../../training/README.md): `.module`, `.compute_loss(batch)`, `.compute_score_batch(batch)`.

## Multivariate strategy

Per-channel-then-max (locked Phase 3 decision):

- For each (B, T, C) input, treat C channels as independent forecasting problems in the batch dimension.
- Reshape to (B·C, T, 1) → forecast → reshape back to (B, T, C).
- Score per step = max_c residual.

Moirai is the exception: it's multivariate-native, so it processes channels jointly.

## LoRA fine-tuning

Enabled by setting `trainer.extra.use_lora=True` (plus optional `lora_r`, `lora_alpha`, `lora_targets`). The Fabric loop calls `dgx_ts_lab.training.peft.wrap_with_lora(detector.module, cfg)` before optimizer setup. Only LoRA params receive gradients.

Default target modules are `("q", "v")` — T5 attention Q and V projections. For other architectures (e.g., Llama-style), override `lora_targets` in the trainer config.

The Fabric loop records `lora_info` (trainable param count, fraction) in `FitResult.metadata` for diagnostics.

## Weight provisioning

See [`docs/foundation_model_provisioning.md`](../../../../../../docs/foundation_model_provisioning.md). Short version:

- **Dev/Windows**: drop HF snapshots into `data/models/<org>/<model>/`.
- **DGX/prod**: run `scripts/setup_mlflow_registry.sh`, then `scripts/register_foundation_models.py` once after sneakernetting weights. Configs use `models:/<name>/Production`.

## Phase 3 caveats

1. **MOMENT** is functional with HF transformers but the architecture deviates slightly from the official MOMENT — for paper-faithful behavior install `momentfm` and we can swap the backbone.
2. **Moirai** is a shell. Real implementation requires `uni2ts` + Moirai-specific patching. Phase 3.5 will fill it in.
3. **Chronos tokenization** uses a basic quantile binning. The real Chronos uses a learned mean-scaling tokenizer — close, but not identical. Sufficient for the bake-off ranking.
