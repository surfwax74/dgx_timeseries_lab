# packages/

uv workspace members. Two packages with deliberately different dependency surfaces.

| Package | Purpose | Heavy deps? |
|---|---|---|
| [`dgx_ts_core/`](dgx_ts_core/README.md) | Pure interfaces (Protocols, dataclasses, registries). | No — numpy only. |
| [`dgx_ts_lab/`](dgx_ts_lab/README.md) | Implementations: datasets, detectors, trainers, MLflow, CLI. | Yes — torch, lightning, mlflow, hydra. |

Downstream MLOps systems (`mm_mlops` etc.) install only `dgx_ts_core` to consume exported artifacts — they never need the training stack.

## Layout

Each package is standard `src/` layout:

```
<package>/
  pyproject.toml
  README.md           ← package overview
  src/<package>/
    ...               ← module tree, each dir has its own README
  tests/
    README.md
    test_*.py
```
