# dgx_ts_core/tests

Unit tests for the pure interfaces. None of these tests may import torch, lightning, mlflow, or hydra — the `test_no_torch_dep.py` test enforces this invariant.

## Files

| File | Tests |
|---|---|
| `test_window.py` | `TelemetryWindow` construction, validation, immutability. |
| `test_capabilities.py` | `Capabilities`, `FitMode`, `OutputKind`. |
| `test_registry.py` | `Registry` register/get/duplicate-key behavior. |
| `test_no_torch_dep.py` | **Critical invariant**: importing any `dgx_ts_core` module must not load torch/lightning/mlflow/hydra. |

Run with `uv run pytest packages/dgx_ts_core/`.
