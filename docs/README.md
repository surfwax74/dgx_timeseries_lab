# docs/

Cross-cutting documentation. Per-module documentation lives next to the code as `README.md` files — start at the [repo root README](../README.md) and walk the tree.

## Files

| File | Read when… |
|---|---|
| [`intern_onboarding.md`](intern_onboarding.md) | **First time using this lab.** Step-by-step hands-on walkthrough — 8 levels from "run the smoke test" up to "design and present your own bake-off". Includes per-level checkpoints. Bookmark this for new contributors. |
| [`experiments_cookbook.md`](experiments_cookbook.md) | **You need the exact command for a phase / experiment / tier.** Quick-reference table + per-phase recipes + Hydra override cheat-sheet. **Start here for "how do I run X?"**|
| [`../packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md) | **You need to know what algorithm each detector implements.** 16 detectors + 3 task heads, organized by algorithm family (classical / from-scratch transformers / foundation models / behavior / physics-informed). Includes capabilities matrix and "when to use what" cheat-sheet. |
| [`architecture.md`](architecture.md) | First time touching the codebase. Explains how the pieces fit. |
| [`adding_a_dataset.md`](adding_a_dataset.md) | You want a new `TelemetryDataset` implementation. |
| [`adding_a_model.md`](adding_a_model.md) | You want a new `AnomalyDetector` implementation. |
| [`air_gapped_setup.md`](air_gapped_setup.md) | Setting up the DGX (no internet). Covers NASA dataset provisioning. |
| [`foundation_model_provisioning.md`](foundation_model_provisioning.md) | Provisioning Chronos/MOMENT/Moirai weights — dev (data/models/) and DGX (MLflow Registry) paths. |
| [`foundation_model_roadmap.md`](foundation_model_roadmap.md) | **Backlog for next-phase model additions** — TimesFM, TTM, TimeMoE (TS foundations), Granite-Code + Phi-4 (LLM co-pilot), Sparse Autoencoders (Phase 7 interpretability), ETS classical baseline. HF repos, sizes, integration effort, air-gap sneakernet plan. |
| [`forecasting_rul_bakeoff.md`](forecasting_rul_bakeoff.md) | **Scoping doc for a second bake-off** — distinct from the AD bake-off. Forecasting + RUL leaderboard covering Prophet, ETS, Chronos, Moirai, MOMENT, TimesFM, TTM, and Sat-TSFM-multi-task. Two demo scenarios (battery degradation, fuel projection). ~1 week effort, 5 tasks scoped. |
| [`lift_to_mlops.md`](lift_to_mlops.md) | You're exporting a trained detector for `mm_mlops` to consume. |
| [`llm_ops_copilot.md`](llm_ops_copilot.md) | Setting up / operating the Phase 11 LLM ops co-pilot (4 backends, RAG, tools, procedure synthesis). |
| [`serving_deployment.md`](serving_deployment.md) | Deploying detectors via Triton (exported ONNX bundles). |

## Deployment playbooks

Per-tier walkthroughs (prereqs, install, verify, smoke commands, common issues):

- [`deployment/README.md`](deployment/README.md) — tier overview
- [`deployment/cpu_only.md`](deployment/cpu_only.md)
- [`deployment/rtx3080_workstation.md`](deployment/rtx3080_workstation.md)
- [`deployment/a5000_server.md`](deployment/a5000_server.md)
- [`deployment/dgx_h200.md`](deployment/dgx_h200.md)
- [`deployment/hardware_compatibility_matrix.md`](deployment/hardware_compatibility_matrix.md) — which model fits where

## Phase plans (forward-looking)

- [`phase_plans/README.md`](phase_plans/README.md) — index of multi-phase plans
- [`phase_plans/phases_6_through_11.md`](phase_plans/phases_6_through_11.md) — A1–D13 demonstration extensions (multi-task heads, explanation, cyber AD, PINN, multi-modal, LLM co-pilot)
