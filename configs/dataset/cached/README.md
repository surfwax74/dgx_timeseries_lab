# `configs/dataset/cached/` — read-from-disk aliases

Thin wrapper YAMLs that route `dataset=cached/<name>` to the
`parquet_telemetry` loader with the right `data_path`. Pair with
`scripts/build_dataset.{ps1,sh}` for a "materialize once, reuse forever"
workflow.

## Why this exists

The two dataset patterns:

| Selector | What happens |
|---|---|
| `dataset=presets/leo_eps_24h` | **Regenerates in-memory** every call from the recipe. Fast for small presets, slow for large ones, non-deterministic if the preset spec changes. |
| `dataset=cached/leo_eps_24h` | **Reads from `data/synth/leo_eps_24h/`**. Instant load, byte-identical across runs, needs a one-time `synth` first. |

Use `presets/` while you're iterating on the recipe, `cached/` once
you've locked it in.

## Adding a new cached alias

Two-file recipe:

1. **The preset generator** (already exists or you author it):
   `configs/dataset/presets/<your_name>.yaml` with `_target_key: layered_synth`
   and full channel + component list.

2. **The cached alias** (this directory):
   `configs/dataset/cached/<your_name>.yaml` with just:
   ```yaml
   _target_key: parquet_telemetry
   data_path: data/synth/<your_name>
   ```

Then materialize once:
```
pwsh scripts/build_dataset.ps1 <your_name>
```

And use it in any experiment via `dataset=cached/<your_name>` or in a
benchmark suite entry:
```yaml
- {key: parquet_telemetry, params: {data_path: data/synth/<your_name>}}
```

## Currently cached

| Alias | Preset source | Size on disk | Rebuild time |
|---|---|---:|---:|
| `cached/trivial_synth` | `dataset=trivial_synth` (top-level, not a preset) | ~1 MB | ~5 s |
| `cached/leo_eps_24h` | `presets/leo_eps_24h` (6 ch × 24 h) | ~10 MB | ~2 min |
| `cached/leo_eps_full_24h` | `presets/leo_eps_full_24h` (83 ch × 24 h) | ~200 MB | ~15 min |

## Staleness — this is on you

The cache does NOT track whether the source preset has been edited
since the last materialization. If you tweak `presets/leo_eps_24h.yaml`
and forget to re-materialize, `dataset=cached/leo_eps_24h` will
happily load the stale bytes.

Two ways to handle it:

**Belt-and-suspenders**: run `build_dataset.ps1 <name> --force` before
any important benchmark to guarantee fresh cache.

**Manifest inspection**: every materialized dataset writes
`data/synth/<name>/manifest.yaml` with the `source_config` used to
generate it. Diff that against `configs/dataset/presets/<name>.yaml`
to detect drift:
```
diff (cat data/synth/leo_eps_24h/manifest.yaml) (cat configs/dataset/presets/leo_eps_24h.yaml)
```
