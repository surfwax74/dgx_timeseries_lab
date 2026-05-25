"""One-time helper: register sneakernetted foundation-model weights with MLflow.

Walks ``data/models/<org>/<model>/`` and registers each as an MLflow model
named ``<model>`` (with the ``Production`` stage tag) so the loader can
resolve them via ``models:/<model>/Production``.

Run AFTER you've sneakernetted weights into ``data/models/`` and started
the Registry server (see scripts/setup_mlflow_registry.sh).

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    uv run python scripts/register_foundation_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "data" / "models"


def main() -> int:
    if "MLFLOW_TRACKING_URI" not in os.environ:
        print(
            "ERROR: MLFLOW_TRACKING_URI is not set. Start the Registry first "
            "with scripts/setup_mlflow_registry.sh and export the URI.",
            file=sys.stderr,
        )
        return 1
    if not MODELS_DIR.exists():
        print(f"ERROR: {MODELS_DIR} does not exist. Sneakernet weights first.")
        return 1

    import mlflow

    # Find every <org>/<model>/ subdir that looks like a HF snapshot
    # (must contain config.json).
    found = []
    for org_dir in MODELS_DIR.iterdir():
        if not org_dir.is_dir():
            continue
        for model_dir in org_dir.iterdir():
            if not model_dir.is_dir():
                continue
            if (model_dir / "config.json").exists():
                found.append((org_dir.name, model_dir.name, model_dir))

    if not found:
        print(f"No HF model snapshots found under {MODELS_DIR}. Nothing to register.")
        return 0

    print(f"Found {len(found)} model snapshots to register:")
    for org, name, _ in found:
        print(f"  - {org}/{name}")

    for org, name, model_dir in found:
        artifact_uri = f"models/{org}/{name}"
        with mlflow.start_run(run_name=f"register_{org}_{name}"):
            mlflow.log_artifacts(str(model_dir), artifact_path=artifact_uri)
            run = mlflow.active_run()
            assert run is not None
            model_uri = f"runs:/{run.info.run_id}/{artifact_uri}"
            registered = mlflow.register_model(model_uri=model_uri, name=name)
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=name,
                version=registered.version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(
                f"  Registered models:/{name}/Production "
                f"(version {registered.version})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
