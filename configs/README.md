# configs/

Hydra configuration tree. Run `dgx-ts train experiment=phase0_smoke` (or any other experiment file) to compose a run.

## Layout

```
configs/
├── config.yaml          ← top-level composition + defaults
├── dataset/             ← swap dataset implementations
│   └── presets/         ← canonical pre-built combos
├── model/               ← swap detector implementations
├── trainer/             ← swap Lightning strategy / device / precision
└── experiment/          ← compositions: pick one dataset × model × trainer + tweaks
```

Each subdir has its own README explaining what's inside and how to add a new entry.

## The composition pattern

- A top-level `config.yaml` declares `defaults` and shared keys.
- An `experiment/<name>.yaml` is the most common entry point — it overrides the defaults for a specific run.
- Datasets / models / trainers in the registry are referenced by `_target_key: <registry_key>`.
- Hydra command-line overrides work as usual: `dgx-ts train experiment=phase0_smoke trainer.max_epochs=5 dataset.seed=42`.

## See also

- Subdirs: [`dataset/`](dataset/README.md), [`model/`](model/README.md), [`trainer/`](trainer/README.md), [`experiment/`](experiment/README.md)
- CLI: [`packages/dgx_ts_lab/src/dgx_ts_lab/cli/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/cli/README.md)
