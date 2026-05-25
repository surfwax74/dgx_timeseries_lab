# dgx_ts_lab.evaluation

Concrete metrics and threshold calibration. Standard scikit-learn-backed AD metrics today; point-adjusted F1 and VUS arrive in Phase 2 alongside the bake-off CLI.

## Files

| File | What it provides |
|---|---|
| `metrics.py` | `basic_metrics(y_true, scores, threshold)` → dict with precision/recall/F1/ROC-AUC/PR-AUC. `calibrate_threshold(scores, method, percentile)` → float. |

## Planned (Phase 2)

| File | Adds |
|---|---|
| `benchmark.py` | Orchestrator: runs N detectors × M datasets, writes a comparison report. |
| `point_adjust.py` | Point-adjusted F1 (standard for time-series AD with collective anomalies). |
| `vus.py` | Volume Under the Surface (Paparrizos et al., 2022). |

## Conventions

- All metrics are threshold-independent (AUC) OR take threshold as an explicit arg. Never magic.
- ROC-AUC is the headline ranking metric. F1 is reported but read F1 numbers with the threshold context they live in.
