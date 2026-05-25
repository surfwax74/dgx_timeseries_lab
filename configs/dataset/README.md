# configs/dataset/

Hydra group for dataset selection. Each YAML names a registered `_target_key` and provides its kwargs.

## Files

| YAML | Registry key | What it generates |
|---|---|---|
| `trivial_synth.yaml` | `trivial_synth` | Phase 0 sine + spike. ~10k samples, runs in ms. |
| `layered_synth.yaml` | `layered_synth` | Minimal placeholder for the layered generator — override channels/components inline or use a preset. |
| `presets/` | various | Canonical multi-channel multi-fault presets. See [`presets/README.md`](presets/README.md). |
| `smap.yaml` | `nasa_smap_channel` *(Phase 1)* | Single channel from NASA SMAP — set `channel_id` and `data_root`. |
| `msl.yaml` | `nasa_msl_channel` *(Phase 1)* | Single channel from NASA MSL. |
| `parquet.yaml` | `parquet_telemetry` *(Phase 1)* | Load a pre-generated dataset from a parquet directory. |

## Adding a new dataset config

1. Implement and register the dataset class (see [`packages/dgx_ts_lab/src/dgx_ts_lab/datasets/README.md`](../../packages/dgx_ts_lab/src/dgx_ts_lab/datasets/README.md)).
2. Add a YAML here with `_target_key: <your_key>` + your kwargs.
3. Reference it from an experiment: `dgx-ts train dataset=<your_yaml_name>`.

## Convention

- Top-level keys map 1-to-1 to the registered factory's kwargs.
- Use `_target_:` (Hydra-native) only for nested instances (e.g., `layered_synth.components[]`).
- Use `_target_key:` (our convention) only for the registry-resolved factory.
