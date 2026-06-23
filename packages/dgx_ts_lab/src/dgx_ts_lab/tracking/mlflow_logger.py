"""MLflow tracking integration.

Self-hosted, file-backed: ``mlruns/`` lives at the repo root by default,
making the scaffold air-gap friendly. The same logger transparently swaps
to a remote tracking server when ``MLFLOW_TRACKING_URI`` is set in the env.
"""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any

import mlflow
from dgx_ts_core.models import FitResult


class MLflowLogger(AbstractContextManager["MLflowLogger"]):
    """Thin context-managed wrapper around the MLflow Python API."""

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tags = tags or {}
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._active_run = None

    def __enter__(self) -> MLflowLogger:
        self._active_run = mlflow.start_run(run_name=self._run_name, tags=self._tags)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        mlflow.end_run(status="FINISHED" if exc_type is None else "FAILED")

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow needs primitive-ish params; coerce non-primitives to strings.
        coerced = {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in _flatten(params).items()
        }
        mlflow.log_params(coerced)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v == v}  # drop NaN
        if clean:
            mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_fit_result(self, result: FitResult) -> None:
        # Flatten nested val/test metrics into MLflow scalar metrics.
        for split_key in ("val_metrics", "test_metrics"):
            block = result.metadata.get(split_key)
            if isinstance(block, dict):
                self.log_metrics({f"{split_key}.{k}": v for k, v in block.items()})
        if result.final_loss is not None:
            self.log_metrics({"final_loss": result.final_loss})
        self.log_metrics({"n_steps": float(result.n_steps)})

        # Log the full FitResult.metadata as a JSON artifact for traceability.
        meta_path = Path("fit_result.json")
        meta_path.write_text(json.dumps(_jsonify(result.metadata), indent=2))
        try:
            self.log_artifact(meta_path)
        finally:
            meta_path.unlink(missing_ok=True)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
