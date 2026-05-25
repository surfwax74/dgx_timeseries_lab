# dgx_ts_lab.tracking

MLflow integration. Self-hosted, file-backed by default — air-gap friendly.

## Files

| File | What it provides |
|---|---|
| `mlflow_logger.py` | `MLflowLogger` context manager — wraps `mlflow.start_run` and exposes `log_params`, `log_metrics`, `log_artifact`, `log_fit_result`. |

## How it stores runs

By default, `MLFLOW_TRACKING_URI` is unset → MLflow writes to `mlruns/` at the current working directory (which is the Hydra-managed `outputs/<date>/<time>/` per-run dir, with the artifact backend inside that).

To use a central server, set `MLFLOW_TRACKING_URI=http://<host>:<port>` in the env before launching. The logger picks it up.

## What gets logged per run

- All Hydra params (flattened, primitives coerced to str).
- `final_loss`, `n_steps`, and every key in `result.metadata["val_metrics"]` / `result.metadata["test_metrics"]`.
- Full `result.metadata` as `fit_result.json` artifact.
- The trained detector checkpoint as the `detector/<name>.<ext>` artifact.
