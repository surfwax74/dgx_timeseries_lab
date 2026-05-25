# configs/experiment/

Hydra group for full experiment compositions — one YAML per benchmark / smoke test / scheduled run. Each picks a (dataset, model, trainer) combination and any overrides.

## Files

| YAML | Composition | Purpose |
|---|---|---|
| `phase0_smoke.yaml` | trivial_synth + rolling_mean + single_cpu | Phase 0 acceptance: prove the scaffold runs end-to-end. |
| `phase1_layered.yaml` | presets/leo_eps_24h + rolling_mean + single_cpu | Phase 1 acceptance: prove the layered generator works through the full pipeline. |
| `phase2_bakeoff.yaml` *(planned)* | matrix sweep over from-scratch detectors × datasets | Phase 2 bake-off. |
| `benchmark_suite.yaml` *(planned)* | full leaderboard run | Standing benchmark — invoked by the scheduled-runs skill. |

## How to invoke

```powershell
dgx-ts train experiment=phase1_layered
```

## How to override on the fly

Any field on the composed config can be overridden from the command line:

```powershell
dgx-ts train experiment=phase1_layered dataset.seed=99 trainer.max_epochs=3 mlflow.run_name=custom_run
```

## Convention for writing experiment YAMLs

```yaml
# @package _global_
defaults:
  - override /dataset: <dataset-name>
  - override /model: <model-name>
  - override /trainer: <trainer-name>

mode: zeroshot | finetune | pretrain

mlflow:
  experiment_name: <bucket>
  run_name: <readable-name>
```
