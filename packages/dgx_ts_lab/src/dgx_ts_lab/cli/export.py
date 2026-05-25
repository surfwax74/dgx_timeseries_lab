"""``dgx-ts export`` — emit ONNX + model_card + feature_schema + Triton config.

Phase 5 lift contract. Run after a successful training run on the same
machine that holds the checkpoint.

Usage:

    uv run --no-sync dgx-ts export \
        model=patchtst_mae \
        dataset=parquet \
        checkpoint=checkpoints/patchtst_mae_best.pt \
        threshold=3.14 \
        +output_dir=exported/patchtst_mae \
        +write_triton=true
"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import dgx_ts_lab  # noqa: F401  registrations
from dgx_ts_core.registry import DATASET_REGISTRY, DETECTOR_REGISTRY

from ..serving import (
    export_detector,
    write_feature_schema,
    write_model_card,
    write_triton_ensemble,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_CONFIG_DIR = _REPO_ROOT / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def run(cfg: DictConfig) -> None:
    # ── Required overrides ───────────────────────────────────────────
    checkpoint_raw = cfg.get("checkpoint")
    if not checkpoint_raw:
        raise SystemExit(
            "dgx-ts export requires +checkpoint=<path> — the trained detector to export."
        )
    ckpt_path = Path(checkpoint_raw)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(get_original_cwd()) / ckpt_path
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    threshold = cfg.get("threshold", None)
    if threshold is not None:
        threshold = float(threshold)

    output_dir_raw = Path(cfg.get("output_dir", "exported"))
    output_dir = (
        output_dir_raw
        if output_dir_raw.is_absolute()
        else Path(get_original_cwd()) / output_dir_raw
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    emit_threshold_variant = bool(cfg.get("emit_threshold_variant", True))
    write_triton = bool(cfg.get("write_triton", False))
    triton_store_raw = Path(cfg.get("triton_store", "triton_models"))
    triton_store = (
        triton_store_raw
        if triton_store_raw.is_absolute()
        else Path(get_original_cwd()) / triton_store_raw
    )

    # ── Build detector + load checkpoint ─────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg, dict)
    model_key = model_cfg.pop("_target_key")
    # Use the detector's load() to reconstruct from checkpoint
    detector_cls = DETECTOR_REGISTRY.get(model_key).__wrapped__ if hasattr(
        DETECTOR_REGISTRY.get(model_key), "__wrapped__"
    ) else type(DETECTOR_REGISTRY.create(model_key, **model_cfg))
    # Simpler: build, then load_state_dict-style restore via classmethod load()
    detector_factory = DETECTOR_REGISTRY.get(model_key)
    detector_for_class = detector_factory(**model_cfg)
    detector = type(detector_for_class).load(ckpt_path)

    # ── Build dataset to get channels + stats for the feature schema ──
    ds_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    assert isinstance(ds_cfg, dict)
    ds_key = ds_cfg.pop("_target_key")
    dataset = DATASET_REGISTRY.create(ds_key, **ds_cfg)

    n_channels = len(dataset.channels)
    window_length = int(getattr(detector, "_window_length", None)
                        or detector.capabilities.native_context_len)

    # ── 1. ONNX artifacts ────────────────────────────────────────────
    print(f"==> Exporting ONNX to {output_dir}/")
    onnx_paths = export_detector(
        detector,
        output_dir=output_dir,
        threshold=threshold,
        n_channels=n_channels,
        window_length=window_length,
        emit_threshold_variant=emit_threshold_variant,
    )
    for name, path in onnx_paths.items():
        print(f"  - {name}: {path.relative_to(output_dir.parent)}")

    # ── 2. Model card ────────────────────────────────────────────────
    print(f"==> Writing model_card.yaml")
    write_model_card(
        detector_name=detector.name,
        detector_version=str(cfg.get("version", "0.1.0")),
        capabilities=detector.capabilities,
        intended_subsystem=dataset.subsystem.value,
        training_dataset=dataset.name,
        metrics=dict(cfg.get("metrics", {})),  # caller supplies metrics from the train run
        calibrated_threshold=threshold,
        output_path=output_dir / "model_card.yaml",
        notes=str(cfg.get("notes", "")),
        mlflow_run_id=cfg.get("mlflow_run_id"),
    )

    # ── 3. Feature schema ────────────────────────────────────────────
    print(f"==> Writing feature_schema.yaml")
    write_feature_schema(
        channels=dataset.channels,
        sample_rate_hz=dataset.sample_rate_hz,
        window_length=window_length,
        stats=dataset.stats(),
        output_path=output_dir / "feature_schema.yaml",
    )

    # ── 4. (Optional) Triton model store ─────────────────────────────
    if write_triton:
        print(f"==> Writing Triton model store to {triton_store}/")
        triton_dirs = write_triton_ensemble(
            model_name=detector.name,
            onnx_paths=onnx_paths,
            triton_store=triton_store,
            n_channels=n_channels,
            window_length=window_length,
            max_batch_size=int(cfg.get("triton_max_batch", 16)),
        )
        for name, path in triton_dirs.items():
            print(f"  - {name}: {path}")

    print("\n-- Export complete --")
    print(f"output_dir:        {output_dir}")
    print(f"artifacts:         {list(onnx_paths.keys()) + ['model_card.yaml', 'feature_schema.yaml']}")
    if write_triton:
        print(f"triton_store:      {triton_store}")


if __name__ == "__main__":
    run()
