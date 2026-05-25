# dgx_ts_lab.datasets

Concrete `TelemetryDataset` implementations. Importing this package self-registers every dataset with `dgx_ts_core.registry.DATASET_REGISTRY`.

## Subdirs and files

| Path | What it is | Registry key(s) |
|---|---|---|
| [`synthetic/trivial.py`](synthetic/README.md) | Phase 0 sine + spike smoke dataset | `trivial_synth` |
| [`synthetic/layered/`](synthetic/layered/README.md) | Phase 1 composable L1–L6 generator | `layered_synth` |
| `nasa_telemanom.py` | NASA SMAP / MSL benchmark loader (air-gap aware) | `nasa_smap_channel`, `nasa_msl_channel` |
| `parquet_telemetry.py` | Reads parquet datasets written by `dgx-ts synth` | `parquet_telemetry` |

## Adding a new dataset

1. Implement the `TelemetryDataset` Protocol from `dgx_ts_core.data`.
2. Decorate a factory with `@DATASET_REGISTRY.register("my_key")`.
3. Add `from . import my_module` to this `__init__.py` so importing the package triggers registration.
4. Write `configs/dataset/my_key.yaml` with the `_target_key: my_key` line + your params.
5. Add tests under `packages/dgx_ts_lab/tests/`.

See [`docs/adding_a_dataset.md`](../../../../../docs/adding_a_dataset.md) for a walkthrough.

## Air-gap notes

- **NASA loaders** never auto-download. They expect files at a `data_root` path you supply; the loader prints a clear error pointing at the Telemanom GitHub mirror if files are missing.
- The synth → parquet → parquet_telemetry round trip is the canonical air-gapped workflow: generate once on any machine, distribute the parquet files, load them anywhere.
