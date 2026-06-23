"""Write a ModelCard dataclass to YAML.

The exact field set is defined in ``dgx_ts_core.export.model_card.ModelCard``
so downstream consumers can deserialize without depending on dgx_ts_lab.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dgx_ts_core.export import ModelCard
from dgx_ts_core.models import Capabilities


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def write_model_card(
    *,
    detector_name: str,
    detector_version: str,
    capabilities: Capabilities,
    intended_subsystem: str,
    training_dataset: str,
    metrics: dict[str, float],
    calibrated_threshold: float | None,
    output_path: Path,
    notes: str = "",
    extra: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    mlflow_run_id: str | None = None,
    dgx_ts_lab_version: str | None = None,
) -> ModelCard:
    """Build a ModelCard and write it to ``output_path`` as YAML.

    Returns the in-memory ``ModelCard`` for further use.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[5]
    if dgx_ts_lab_version is None:
        try:
            import dgx_ts_lab

            dgx_ts_lab_version = dgx_ts_lab.__version__
        except Exception:
            dgx_ts_lab_version = "unknown"

    full_extra: dict[str, Any] = dict(extra or {})
    full_extra.setdefault("exported_at", datetime.utcnow().isoformat() + "Z")
    full_extra.setdefault("python_version", platform.python_version())
    full_extra.setdefault("platform", platform.platform())
    full_extra.setdefault("git_sha", _git_sha(repo_root))
    full_extra.setdefault("dgx_ts_lab_version", dgx_ts_lab_version)
    if mlflow_run_id is not None:
        full_extra.setdefault("mlflow_run_id", mlflow_run_id)

    card = ModelCard(
        detector_name=detector_name,
        detector_version=detector_version,
        capabilities=capabilities,
        intended_subsystem=intended_subsystem,
        training_dataset=training_dataset,
        metrics=metrics,
        calibrated_threshold=calibrated_threshold,
        notes=notes,
        extra=full_extra,
    )

    # Serialize. Capabilities is a frozen dataclass; convert to plain dict.
    card_dict = asdict(card)
    # Convert enum value inside capabilities → str
    cap_dict = card_dict["capabilities"]
    if hasattr(cap_dict.get("output_kind"), "value"):
        cap_dict["output_kind"] = cap_dict["output_kind"].value
    elif hasattr(capabilities.output_kind, "value"):
        cap_dict["output_kind"] = capabilities.output_kind.value

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(card_dict, sort_keys=False))
    return card
