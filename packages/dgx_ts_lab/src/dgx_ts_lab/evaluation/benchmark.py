"""Bake-off orchestrator — sweep (detector, dataset, seed) combos and
produce a comparison report.

Called by ``dgx-ts benchmark``. Writes:

    benchmark_report.md     Markdown leaderboard sorted by val ROC-AUC
    benchmark_report.json   Machine-readable per-run metrics
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from dgx_ts_core.models import FitMode
from dgx_ts_core.registry import DATASET_REGISTRY, DETECTOR_REGISTRY, TRAINER_REGISTRY
from dgx_ts_core.training import TrainConfig


@dataclass
class BenchmarkRun:
    detector_key: str
    dataset_key: str
    seed: int
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0
    error: str | None = None
    notes: str = ""


def run_benchmark(
    suite_cfg: DictConfig,
    base_trainer_cfg: DictConfig,
    output_dir: Path,
) -> list[BenchmarkRun]:
    """Execute the cartesian product of detectors × datasets × seeds.

    `suite_cfg` shape:
        detectors: list of {key, params: {...}}
        datasets:  list of {key, params: {...}}
        seeds:     list of int
        mode:      pretrain | finetune | zeroshot

    `base_trainer_cfg` is the trainer YAML dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[BenchmarkRun] = []

    detectors = list(suite_cfg.detectors)
    datasets = list(suite_cfg.datasets)
    seeds = list(suite_cfg.get("seeds", [0]))
    mode_str = str(suite_cfg.get("mode", "zeroshot"))

    trainer_cfg = OmegaConf.to_container(base_trainer_cfg, resolve=True)
    assert isinstance(trainer_cfg, dict)
    trainer_key = trainer_cfg.pop("_target_key")
    trainer = TRAINER_REGISTRY.create(trainer_key)

    for dataset_entry in datasets:
        ds_key = dataset_entry["key"]
        ds_params = dict(dataset_entry.get("params", {}))
        for detector_entry in detectors:
            det_key = detector_entry["key"]
            det_params = dict(detector_entry.get("params", {}))
            for seed in seeds:
                run = BenchmarkRun(detector_key=det_key, dataset_key=ds_key, seed=int(seed))
                try:
                    t0 = time.time()
                    # Build dataset (seed override for reproducibility)
                    ds_kwargs = {**ds_params, "seed": int(seed)} if "seed" in ds_params else ds_params
                    dataset = DATASET_REGISTRY.create(ds_key, **ds_kwargs)
                    detector = DETECTOR_REGISTRY.create(det_key, **det_params)

                    cfg_dict = {
                        **trainer_cfg,
                        "seed": int(seed),
                        "checkpoint_dir": Path(trainer_cfg.get("checkpoint_dir", "checkpoints"))
                        / f"bench_{det_key}_{ds_key}_s{seed}",
                    }
                    train_cfg = TrainConfig(**cfg_dict)

                    result = trainer.fit(detector, dataset, FitMode(mode_str), train_cfg)
                    run.val_metrics = dict(result.metadata.get("val_metrics", {}))
                    run.test_metrics = dict(result.metadata.get("test_metrics", {}))
                    run.elapsed_s = time.time() - t0
                    # Persist raw (scores, labels) so `dgx-ts viz` can rebuild
                    # ROC / PR curves later without re-running the model.
                    _save_run_arrays(
                        output_dir,
                        det_key=det_key,
                        ds_key=ds_key,
                        seed=int(seed),
                        arrays=result.metadata,
                    )
                except Exception as e:  # noqa: BLE001
                    run.error = f"{type(e).__name__}: {e}"
                runs.append(run)
                if run.error:
                    summary = f"ERROR: {run.error}"
                else:
                    auc = run.val_metrics.get("roc_auc", float("nan"))
                    summary = f"val_roc_auc={auc:.3f}"
                print(f"  [{det_key} x {ds_key} seed={seed}] {summary}")

    _write_reports(runs, output_dir)
    return runs


def _save_run_arrays(
    output_dir: Path,
    det_key: str,
    ds_key: str,
    seed: int,
    arrays: dict[str, Any],
) -> None:
    """If the trainer surfaced val/test score+label arrays, save them as npz.

    Filename layout: ``{detector}__{dataset}__s{seed}__{split}.npz`` with
    keys ``scores`` and ``labels``. Used by `dgx_ts_lab.evaluation.visualize`.
    """
    for split in ("val", "test"):
        arr = arrays.get(f"{split}_arrays") or {}
        scores = arr.get("scores")
        labels = arr.get("labels")
        if scores is None or labels is None:
            continue
        if scores.size == 0:
            continue
        out_path = output_dir / f"{det_key}__{ds_key}__s{seed}__{split}.npz"
        np.savez(out_path, scores=scores.astype(np.float32), labels=labels.astype(np.bool_))


def _write_reports(runs: list[BenchmarkRun], output_dir: Path) -> None:
    # JSON
    json_path = output_dir / "benchmark_report.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "detector": r.detector_key,
                    "dataset": r.dataset_key,
                    "seed": r.seed,
                    "val_metrics": r.val_metrics,
                    "test_metrics": r.test_metrics,
                    "elapsed_s": r.elapsed_s,
                    "error": r.error,
                }
                for r in runs
            ],
            indent=2,
            default=str,
        )
    )

    # Markdown leaderboard sorted by val ROC-AUC desc, NaN/error to bottom.
    def sort_key(r: BenchmarkRun) -> float:
        v = r.val_metrics.get("roc_auc", float("nan"))
        return -v if v == v else 1e9  # NaN sinks

    ranked = sorted(runs, key=sort_key)
    lines = [
        "# Benchmark report",
        "",
        f"Total runs: {len(runs)} ({sum(1 for r in runs if r.error)} errored)",
        "",
        "| Rank | Detector | Dataset | Seed | val ROC-AUC | val F1 | val PR-AUC | test ROC-AUC | test F1 | time (s) | error |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(ranked, start=1):
        lines.append(
            "| {i} | {det} | {ds} | {seed} | {vra:.3f} | {vf1:.3f} | {vpr:.3f} | {tra:.3f} | {tf1:.3f} | {t:.1f} | {err} |".format(
                i=i,
                det=r.detector_key,
                ds=r.dataset_key,
                seed=r.seed,
                vra=r.val_metrics.get("roc_auc", float("nan")),
                vf1=r.val_metrics.get("f1", float("nan")),
                vpr=r.val_metrics.get("pr_auc", float("nan")),
                tra=r.test_metrics.get("roc_auc", float("nan")),
                tf1=r.test_metrics.get("f1", float("nan")),
                t=r.elapsed_s,
                err=r.error or "",
            )
        )
    (output_dir / "benchmark_report.md").write_text("\n".join(lines))
