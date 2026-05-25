# dgx_ts_lab/tests

Unit + integration tests for implementations.

## Files

| File | Tests |
|---|---|
| `test_trivial_synth.py` | Phase 0 trivial sine+spike dataset. |
| `test_rolling_mean_detector.py` | `RollingMeanDetector` fit/score/save/load + ROC-AUC against injected spikes. |
| `test_lightning_trainer.py` | End-to-end trainer.fit() classical-detector path with val/test metric calculation. |
| `test_registry_wired.py` | Importing `dgx_ts_lab` registers `trivial_synth`, `layered_synth`, `rolling_mean`, `lightning`, `nasa_smap_channel`, `nasa_msl_channel`, `parquet_telemetry`. |
| `test_layered_components.py` | Every L1–L6 component standalone + end-to-end layered dataset integration test. |
| `test_nasa_telemanom.py` | NASA loader against a synthetic fixture (we don't ship the real NASA files). |
| `test_synth_parquet_roundtrip.py` | `dgx-ts synth` → write → `parquet_telemetry` load → verify identical. |

Run with `uv run pytest packages/dgx_ts_lab/`.
