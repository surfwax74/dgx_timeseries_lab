"""``dgx-ts explain`` — produce explanation reports for anomalous windows.

Hydra-wired. Composes:
    dataset    where to read windows from
    model      detector class (to instantiate before loading checkpoint)
    +checkpoint  path to trained detector checkpoint
    +window_idx  (optional) explain this specific window
    +top_k       (default 5) explain top-K most anomalous windows from a scan
    +output_dir  where to write reports (default: outputs/explanations/)
    +coupling_strategy  auto | declared | learned | none

Writes ``explanation_<idx>.md`` + ``explanation_<idx>.json`` per window to
``output_dir/``, plus optional PNG visualizations.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from dgx_ts_core.registry import DATASET_REGISTRY, DETECTOR_REGISTRY
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import dgx_ts_lab  # noqa: F401  registrations

from ..explanation import (
    ExplanationReport,
    attribute_window,
    build_coupling_graph,
    walk_cascade,
    write_report,
)
from ..explanation.visualize import (
    render_channel_attribution,
    render_score_timeline,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_CONFIG_DIR = _REPO_ROOT / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIG_DIR), config_name="config")
def run(cfg: DictConfig) -> None:
    checkpoint_raw = cfg.get("checkpoint")
    if not checkpoint_raw:
        raise SystemExit(
            "dgx-ts explain requires +checkpoint=<path> — the trained detector to explain."
        )
    ckpt_path = Path(checkpoint_raw)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(get_original_cwd()) / ckpt_path
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    window_idx = cfg.get("window_idx", None)
    top_k = int(cfg.get("top_k", 5))
    output_dir_raw = Path(cfg.get("output_dir", "outputs/explanations"))
    output_dir = (
        output_dir_raw
        if output_dir_raw.is_absolute()
        else Path(get_original_cwd()) / output_dir_raw
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    coupling_strategy = str(cfg.get("coupling_strategy", "auto"))
    render_plots = bool(cfg.get("render_plots", True))

    # ── Build detector from registry + load checkpoint ──────────────
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg, dict)
    model_key = model_cfg.pop("_target_key")
    proto = DETECTOR_REGISTRY.create(model_key, **model_cfg)
    detector = type(proto).load(ckpt_path)

    # ── Build dataset ────────────────────────────────────────────────
    ds_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    assert isinstance(ds_cfg, dict)
    ds_key = ds_cfg.pop("_target_key")
    dataset = DATASET_REGISTRY.create(ds_key, **ds_cfg)

    # ── Build coupling graph (once, cached) ──────────────────────────
    print(f"==> Building coupling graph (strategy={coupling_strategy})")
    graph = build_coupling_graph(dataset, strategy=coupling_strategy)
    print(f"    graph: {graph.source}, {graph.n_edges()} edges")

    # ── Decide which window(s) to explain ───────────────────────────
    window_length = int(getattr(detector, "_window_length", None)
                        or detector.capabilities.native_context_len)
    window_stride = window_length

    if window_idx is not None:
        target_starts = [int(window_idx)]
    else:
        # Scan all windows, score them, pick top-K
        print(f"==> Scoring windows (length={window_length}, stride={window_stride})")
        scored_windows: list[tuple[int, float]] = []
        for i, w in enumerate(dataset.windows(length=window_length, stride=window_stride)):
            score = float(detector.score(w).scores.max())
            scored_windows.append((i * window_stride, score))
        scored_windows.sort(key=lambda x: -x[1])
        target_starts = [s for s, _ in scored_windows[:top_k]]
        print(f"    selected {len(target_starts)} most-anomalous starts: {target_starts}")

    # ── For each target window, produce an explanation report ───────
    all_scores_for_plot = None
    if render_plots and window_idx is None:
        # Pre-compute the full timeline for the score plot
        all_scores: list[float] = []
        for w in dataset.windows(length=window_length, stride=window_stride):
            all_scores.append(float(detector.score(w).scores.max()))
        all_scores_for_plot = np.asarray(all_scores)

    written = []
    for start in target_starts:
        # We need the window at this specific start — re-walk to find it.
        chosen = None
        for w in dataset.windows(length=window_length, stride=window_stride):
            if int(w.provenance.get("start", -1)) == start:
                chosen = w
                break
        if chosen is None:
            # Re-walk linearly and step to the start
            for w in dataset.windows(length=window_length, stride=1):
                if int(w.provenance.get("start", -1)) == start:
                    chosen = w
                    break
        if chosen is None:
            print(f"    skipping start={start}: window not found")
            continue

        ranked = attribute_window(detector, chosen, top_k=15)
        cascade = []
        if ranked:
            cascade = walk_cascade(graph, target_channel=ranked[0].channel_name, top_k=5)

        plot_paths: dict[str, str] = {}
        if render_plots:
            attr_png = output_dir / f"explanation_{start:06d}_attribution.png"
            render_channel_attribution(ranked, attr_png)
            plot_paths["channel_attribution"] = str(attr_png.relative_to(output_dir))
            if all_scores_for_plot is not None:
                idx_in_scan = start // window_stride
                timeline_png = output_dir / f"explanation_{start:06d}_timeline.png"
                render_score_timeline(
                    all_scores_for_plot,
                    window_idx_range=(idx_in_scan, idx_in_scan + 1),
                    threshold=None,
                    out_path=timeline_png,
                )
                plot_paths["score_timeline"] = str(timeline_png.relative_to(output_dir))

        report = ExplanationReport(
            detector_name=detector.name,
            dataset_name=dataset.name,
            window_idx=start,
            window_length=window_length,
            window_start_ms=int(chosen.timestamps[0]),
            anomaly_score=float(detector.score(chosen).scores.max()),
            ranked_channels=ranked,
            attribution_method=("integrated_gradients"
                                if hasattr(detector, "module") and detector.module is not None
                                else "permutation"),
            cascade=cascade,
            coupling_source=graph.source,
            plot_paths=plot_paths,
        )
        md_path, json_path = write_report(report, output_dir)
        written.append(md_path)
        print(f"  wrote {md_path.name} + .json")

    print(f"\n==> Wrote {len(written)} explanation report(s) to {output_dir}")


if __name__ == "__main__":
    run()
