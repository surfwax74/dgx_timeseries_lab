# experiments/

Numbered, reproducible experiment outputs. Each subdir holds the artifacts (configs snapshot, MLflow run id, generated reports) for one significant experimental finding worth preserving.

## Naming convention

```
NNN_<short-description>/
```

Where `NNN` is a zero-padded sequence number (`001_`, `002_`, …).

## What lives in each experiment dir

- `README.md` — what was done, why, what was found, how to reproduce.
- `config_snapshot/` — the exact Hydra config used (the `.hydra/` folder from the Hydra run).
- `mlflow_run_id.txt` — pointer to the MLflow run (the artifacts live in `mlruns/`, not duplicated here).
- `report.md` or `report.ipynb` — analysis writeup.
- `artifacts/` — derived plots, tables, exported models (large files in `.gitignore`).

## Convention

- One experiment per significant finding, not one per run. Casual exploration goes in `notebooks/` or scratch dirs.
- Experiments are append-only — once committed, don't rewrite history. Add a new experiment dir if conclusions change.
- The Phase 0 / Phase 1 smoke runs live in MLflow under `phase0_smoke` and `phase1_layered` experiment buckets; only graduate them to an `experiments/` dir when there's a finding worth pinning.
