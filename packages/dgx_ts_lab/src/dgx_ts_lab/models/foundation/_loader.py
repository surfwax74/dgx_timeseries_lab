"""Dual-path foundation-model weight loader.

Air-gap strategy: weights live in two possible locations on the DGX.

    1. **MLflow Model Registry** (primary, production):
       URI like ``models:/chronos-t5-small/Production``. Requires a
       running MLflow server (set ``MLFLOW_TRACKING_URI``).

    2. **Local filesystem fallback** (dev, offline):
       ``data/models/<org>/<model_name>/`` containing the snapshot
       (config.json, model.safetensors, tokenizer files, etc.). This is
       what you populate on a Windows dev box without a Registry running.

Resolution order: env says Registry → try Registry → if unreachable or
URI doesn't start with ``models:`` → fall back to local path.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT_ENV = "DGX_TS_LAB_DATA_ROOT"


def _local_root() -> Path:
    """Where ``data/models/`` lives. Override with ``DGX_TS_LAB_DATA_ROOT`` env."""
    override = os.environ.get(REPO_ROOT_ENV)
    if override:
        return Path(override) / "models"
    # Default: repo_root/data/models — works for both dev box and DGX
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").exists() or (parent / ".git").exists():
            return parent / "data" / "models"
    return Path.cwd() / "data" / "models"


def resolve_model_path(name_or_uri: str) -> Path:
    """Resolve a model name/URI to a local path the loader can read from.

    Accepts:
        - ``"models:/<name>/<stage>"``     → fetched from MLflow Registry
        - ``"<org>/<model_name>"``         → looked up under data/models/<org>/<model_name>
        - ``"<plain_name>"``               → looked up under data/models/<plain_name>
        - an absolute or repo-relative path → returned as-is if it exists

    Returns a Path. Raises FileNotFoundError with a clear pointer if neither
    Registry nor local copy is available.
    """
    if not name_or_uri:
        raise ValueError("model name_or_uri is empty")

    # ── 1. Direct path? ──────────────────────────────────────────────
    direct = Path(name_or_uri)
    if direct.exists() and direct.is_dir():
        return direct

    # ── 2. MLflow Registry URI? ──────────────────────────────────────
    if name_or_uri.startswith("models:/"):
        try:
            import mlflow  # local import to keep cold-import fast

            local = mlflow.artifacts.download_artifacts(artifact_uri=name_or_uri)
            return Path(local)
        except Exception as e:  # noqa: BLE001
            # Fall through to local lookup; the Registry might be unreachable
            # (dev box, network blip). Log and continue.
            _local = _local_root() / name_or_uri.removeprefix("models:/").replace("/", os.sep)
            if _local.exists():
                return _local
            raise FileNotFoundError(
                f"MLflow Registry URI '{name_or_uri}' could not be resolved "
                f"({type(e).__name__}: {e}) and no local fallback at {_local}.\n"
                "Provision: see docs/foundation_model_provisioning.md."
            ) from e

    # ── 3. Local under data/models/ ──────────────────────────────────
    local = _local_root() / name_or_uri
    if local.exists() and local.is_dir():
        return local

    raise FileNotFoundError(
        f"Model '{name_or_uri}' not found.\n"
        f"  Checked: {local}\n"
        f"  And: MLflow Registry URI 'models:/{name_or_uri}/Production'\n"
        "Provision: see docs/foundation_model_provisioning.md."
    )
