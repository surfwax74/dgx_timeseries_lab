# dgx_ts_lab.explanation

Phase 7 — detector-agnostic explanation layer. When AD fires, this package turns the raw anomaly score into a structured explanation: which channels drove the score, which UPSTREAM channels are the likely root cause, and a polished Markdown + JSON report.

## Files

| File | Role |
|---|---|
| `report_schema.py` | `ExplanationReport`, `ChannelAttribution`, `Cause` dataclasses — the contract Phase 11 RAG consumes. |
| `coupling_graph.py` | `CouplingGraph` + `build_coupling_graph()` — dual-source: declared from LayeredSyntheticDataset components OR learned from cross-channel Pearson + lagged correlations on training data. |
| `attribution.py` | `attribute_window()` — Captum Integrated Gradients for differentiable detectors, permutation importance fallback for classical ones. |
| `cascade_walker.py` | `walk_cascade()` — BFS up to N hops along the coupling graph; ranks upstream causes by decayed edge weight. |
| `visualize.py` | matplotlib PNG renderers: per-step score timeline + top-K channel attribution bar chart. |
| `report_writer.py` | `write_report()` — emits `explanation_<idx>.md` (human) + `explanation_<idx>.json` (machine-consumable). |

## CLI

```powershell
# Explain a specific window
dgx-ts explain dataset=parquet model=rolling_mean +checkpoint=checkpoints/rolling_mean.npz +window_idx=42

# Or scan and explain the top-K most anomalous windows
dgx-ts explain dataset=parquet model=rolling_mean +checkpoint=checkpoints/rolling_mean.npz +top_k=5
```

Writes to `outputs/explanations/` by default (configurable with `+output_dir=...`).

## Attribution path selection

```
detector has .module + .compute_score_batch?
    ├── yes → try Captum IntegratedGradients (n_steps=32)
    │           ├── success → method="integrated_gradients"
    │           └── failure → fall through
    └── permutation importance (n_trials=5 channel shuffles)
                method="permutation"
```

## Coupling graph sources

| Source | When | What it captures |
|---|---|---|
| `declared` | LayeredSyntheticDataset with `_components` attribute | Ground-truth L3 coupling: `LinearCoupling`, `InverseCoupling`, `SumCoupling`. Gains + lags exact. |
| `learned` | NASA / parquet / any dataset with `_data` | Cross-channel Pearson correlations (optionally lagged). Edges retained where `|corr| > threshold` (default 0.3). |
| `none` | Manual fallback | Empty graph; cascade walker returns no causes. Reports still produce attribution + score. |

`build_coupling_graph(strategy="auto")` (the default) tries declared then falls back to learned. Override with `strategy="learned"` to force the correlation path even on layered_synth (useful for ablation studies).

## What's in the JSON sibling

```json
{
  "detector_name": "patchtst_mae",
  "dataset_name": "leo_eps_full_24h",
  "window_idx": 12800,
  "anomaly_score": 4.231,
  "ranked_channels": [
    {"channel_name": "bus_voltage", "score": 1.000, "rank": 1, "physics_covered": false},
    {"channel_name": "bat_a_voltage", "score": 0.734, "rank": 2, "physics_covered": true}
  ],
  "cascade": [
    {"source_channel": "pdu_a_pri_i", "target_channel": "bus_voltage", "weight": -0.06, "lag_steps": 0, "via": ""}
  ],
  "coupling_source": "declared",
  "fault_type_predicted": null,
  "plot_paths": {
    "channel_attribution": "explanation_012800_attribution.png",
    "score_timeline": "explanation_012800_timeline.png"
  }
}
```

Stable schema — when the Markdown template changes, the JSON doesn't.

## See also

- Phase 7 plan: [`docs/phase_plans/phases_6_through_11.md`](../../../../../../docs/phase_plans/phases_6_through_11.md)
- Cascade walker uses the layered-synth coupling components: [`packages/dgx_ts_lab/src/dgx_ts_lab/datasets/synthetic/layered/coupling.py`](../../datasets/synthetic/layered/coupling.py)
