"""``dgx-ts benchmark`` — run the bake-off over (detector × dataset × seed).

Reads a benchmark-suite YAML (under ``configs/experiment/`` typically),
runs the matrix sweep, writes a Markdown leaderboard + JSON report.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import dgx_ts_lab  # noqa: F401  side-effect: register implementations

from ..evaluation.benchmark import run_benchmark

_REPO_ROOT = Path(__file__).resolve().parents[5]
_CONFIG_DIR = _REPO_ROOT / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def run(cfg: DictConfig) -> None:
    suite = cfg.get("suite")
    if suite is None:
        raise SystemExit(
            "benchmark expects a `suite=<name>` group — e.g., "
            "`dgx-ts benchmark suite=phase2_bakeoff`. See configs/experiment/."
        )

    output_raw = Path(cfg.get("output_dir", "benchmark_reports"))
    output_root = (
        output_raw
        if output_raw.is_absolute()
        else Path(get_original_cwd()) / output_raw
    )
    name = suite.get("name", "unnamed_bench")
    output_dir = output_root / name

    print(f"-- Running benchmark suite '{name}' -> {output_dir}")
    print(OmegaConf.to_yaml(suite))

    runs = run_benchmark(suite, cfg.trainer, output_dir)
    n_err = sum(1 for r in runs if r.error)
    print(
        f"\n-- Benchmark complete: {len(runs)} runs ({n_err} errored). "
        f"Report: {output_dir / 'benchmark_report.md'}"
    )


if __name__ == "__main__":
    run()
