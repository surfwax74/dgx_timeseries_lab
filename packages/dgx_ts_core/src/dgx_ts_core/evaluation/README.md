# dgx_ts_core.evaluation

Evaluation-side contracts: `Metric` Protocol and `EvalReport` result type. Concrete metric implementations live in `dgx_ts_lab.evaluation`.

## Files

| File | What it defines |
|---|---|
| `metrics.py` | `Metric` Protocol — callable with `(y_true, y_pred, scores) → float`. |
| `result.py` | `EvalReport` dataclass — standard structured eval output (per-detector, per-dataset, with `metrics` + optional `per_channel` breakdown). |
| `__init__.py` | Re-exports. |

## Conventions

- All metrics return a single `float`. Multi-value reports go into the `metrics: dict[str, float]` on `EvalReport`.
- `higher_is_better` is a property on each `Metric` so leaderboard code knows which direction to sort.

## See also

- Parent: [`../README.md`](../README.md)
- Concrete metrics: [`packages/dgx_ts_lab/src/dgx_ts_lab/evaluation/README.md`](../../../../dgx_ts_lab/src/dgx_ts_lab/evaluation/README.md)
