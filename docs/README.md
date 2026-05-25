# docs/

Cross-cutting documentation. Per-module documentation lives next to the code as `README.md` files — start at the [repo root README](../README.md) and walk the tree.

## Files

| File | Read when… |
|---|---|
| [`architecture.md`](architecture.md) | First time touching the codebase. Explains how the pieces fit. |
| [`adding_a_dataset.md`](adding_a_dataset.md) | You want a new `TelemetryDataset` implementation. |
| [`adding_a_model.md`](adding_a_model.md) | You want a new `AnomalyDetector` implementation. |
| [`air_gapped_setup.md`](air_gapped_setup.md) | Setting up the DGX (no internet). Covers NASA dataset provisioning. |
| [`foundation_model_provisioning.md`](foundation_model_provisioning.md) | Provisioning Chronos/MOMENT/Moirai weights — dev (data/models/) and DGX (MLflow Registry) paths. |
| [`lift_to_mlops.md`](lift_to_mlops.md) | You're exporting a trained detector for `mm_mlops` to consume. |

## Deployment playbooks

Per-tier walkthroughs (prereqs, install, verify, smoke commands, common issues):

- [`deployment/README.md`](deployment/README.md) — tier overview
- [`deployment/cpu_only.md`](deployment/cpu_only.md)
- [`deployment/rtx3080_workstation.md`](deployment/rtx3080_workstation.md)
- [`deployment/a5000_server.md`](deployment/a5000_server.md)
- [`deployment/dgx_h200.md`](deployment/dgx_h200.md)
- [`deployment/hardware_compatibility_matrix.md`](deployment/hardware_compatibility_matrix.md) — which model fits where
