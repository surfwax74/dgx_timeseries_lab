"""``dgx-ts train`` — fit a detector against a dataset.

Wires the registry-based factory pattern through Hydra: each of dataset /
model / trainer YAMLs declares a ``_target_key`` that names a registered
factory in dgx_ts_core.registry. The CLI pops that key and passes the
remaining YAML fields as kwargs.

Run end-to-end with:
    uv run dgx-ts train experiment=phase0_smoke
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Side-effect imports — trigger self-registration of bundled implementations.
import dgx_ts_lab  # noqa: F401
from dgx_ts_core.models import FitMode
from dgx_ts_core.registry import DATASET_REGISTRY, DETECTOR_REGISTRY, TRAINER_REGISTRY
from dgx_ts_core.training import TrainConfig

from ..tracking import MLflowLogger

# configs/ lives at the repo root. Resolve the path from this file's location:
# packages/dgx_ts_lab/src/dgx_ts_lab/cli/train.py → up 5 to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_CONFIG_DIR = _REPO_ROOT / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # ── Build dataset ────────────────────────────────────────────────────
    ds_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    assert isinstance(ds_cfg, dict)
    ds_key = ds_cfg.pop("_target_key")
    dataset = DATASET_REGISTRY.create(ds_key, **ds_cfg)

    # ── Build detector ───────────────────────────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg, dict)
    model_key = model_cfg.pop("_target_key")
    detector = DETECTOR_REGISTRY.create(model_key, **model_cfg)

    # ── Build trainer + config ───────────────────────────────────────────
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_cfg, dict)
    trainer_key = trainer_cfg.pop("_target_key")
    trainer = TRAINER_REGISTRY.create(trainer_key)
    if "checkpoint_dir" in trainer_cfg:
        trainer_cfg["checkpoint_dir"] = Path(trainer_cfg["checkpoint_dir"])
    train_config = TrainConfig(**trainer_cfg)

    mode = FitMode(cfg.mode)

    # ── Run + log ────────────────────────────────────────────────────────
    mlflow_cfg = cfg.mlflow
    with MLflowLogger(
        experiment_name=mlflow_cfg.experiment_name,
        run_name=mlflow_cfg.run_name,
        tracking_uri=mlflow_cfg.tracking_uri,
    ) as logger:
        logger.log_params(
            {
                "dataset": ds_key,
                "model": model_key,
                "trainer": trainer_key,
                "mode": mode.value,
                "trainer_cfg": {k: str(v) for k, v in trainer_cfg.items()},
            }
        )

        result = trainer.fit(detector, dataset, mode, train_config)
        logger.log_fit_result(result)

        # Persist the detector and ship it as an MLflow artifact.
        train_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = train_config.checkpoint_dir / f"{detector.name}.npz"
        detector.save(ckpt_path)
        logger.log_artifact(ckpt_path, artifact_path="detector")

        # Surface headline numbers in stdout for human review.
        print("\n-- Run summary --")
        print(f"detector:      {detector.name}")
        print(f"dataset:       {dataset.name}")
        print(f"mode:          {mode.value}")
        print(f"threshold:     {result.metadata.get('threshold')}")
        print(f"val_metrics:   {result.metadata.get('val_metrics')}")
        print(f"test_metrics:  {result.metadata.get('test_metrics')}")
        print(f"checkpoint:    {ckpt_path}")


if __name__ == "__main__":
    run()
