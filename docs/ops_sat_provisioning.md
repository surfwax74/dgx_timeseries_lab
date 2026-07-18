# OPS-SAT Provisioning Guide

ESA's OPS-SAT is a flying software-defined satellite testbed. The
publicly released OPS-SAT anomaly-detection dataset gives us real
orbital telemetry — hundreds of channels, weeks of contiguous
coverage — as a complement to the smaller NASA Telemanom benchmark.

**Rough size**: ~10 GB uncompressed, ~100 channels, ~50× the NASA
Telemanom volume.

## Provisioning workflow (connected machine → sneakernet → DGX)

### Step 1 — Download on a connected machine

```powershell
# Windows / Linux — needs internet
cd C:\dev\dgx_timeseries_lab
python scripts\download_ops_sat.py
```

This fetches the dataset zip (~2 GB compressed) into
`data/ops_sat_raw/`. **Verify the URL first** — the ESA / Zenodo host
occasionally rotates records. Check:

- https://opssat.esa.int/
- https://zenodo.org/search?q=OPS-SAT
- https://www.esa.int/Enabling_Support/Space_Engineering_Technology/OPS-SAT

If you need a different URL:

```powershell
python scripts\download_ops_sat.py --url https://new.url.here/file.zip
```

Update `DATASET_URL` in `scripts/download_ops_sat.py` after verifying,
so future users don't hit the same rotation.

### Step 2 — Convert to our parquet layout

```powershell
python scripts\convert_ops_sat_to_parquet.py `
    --raw data\ops_sat_raw `
    --output data\ops_sat
```

Reads the raw CSVs, writes a canonical `data/ops_sat/` directory in the
same format `dgx-ts synth` produces (data.parquet + labels.parquet +
channels.yaml + manifest.yaml + fault_log.json).

**Schema adaptivity**: the converter tries a small set of common column
names (`timestamp`, `time`, `utc` for the time column; `start_time`,
`begin` for anomaly starts; etc.). If the release you downloaded uses
different names, the converter fails loudly and you edit the constants
at the top of `scripts/convert_ops_sat_to_parquet.py` to match.

Optional flags for corpus tuning:

```powershell
# Downsample to save disk (1 Hz → 0.1 Hz)
python scripts\convert_ops_sat_to_parquet.py --raw data\ops_sat_raw --downsample 10

# Keep only a subset of channels (~10× smaller output)
python scripts\convert_ops_sat_to_parquet.py --raw data\ops_sat_raw --channels channel_45,channel_46,channel_47
```

### Step 3 — Verify the loader can read it

```powershell
$env:UV_NO_SYNC = '1'
.\.venv\Scripts\python.exe -c @"
from dgx_ts_lab.datasets.parquet_telemetry import ParquetTelemetryDataset
ds = ParquetTelemetryDataset(data_path='data/ops_sat')
print('Loaded:', ds.name)
print('Channels:', len(ds.channels))
print('Samples:', len(ds._data))
print('Anomaly fraction:', float(ds._labels.mean()))
"@
```

Or via Hydra:

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main train `
    dataset=cached/ops_sat `
    model=patchtst_mae `
    trainer=cpu trainer.max_epochs=1
```

### Step 4 — Sneakernet to the DGX (if air-gapped)

```bash
tar czf ops_sat.tar.gz data/ops_sat/
sha256sum ops_sat.tar.gz > ops_sat.tar.gz.sha256
# Transfer both files to the DGX, then:
sha256sum -c ops_sat.tar.gz.sha256
tar xzf ops_sat.tar.gz
```

Identical `dataset=cached/ops_sat` command works on the DGX now.

## Using OPS-SAT in a bake-off

Same pattern as any other cached dataset. Add to a benchmark suite:

```yaml
suite:
  datasets:
    - {key: parquet_telemetry, params: {data_path: data/nasa_smap}}     # if you have the loader
    - {key: parquet_telemetry, params: {data_path: data/ops_sat}}       # OPS-SAT via parquet
    - {key: parquet_telemetry, params: {data_path: data/synth/leo_eps_24h}}
```

Or in a `dgx-ts train` for a single model:

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main train `
    dataset=cached/ops_sat `
    model=patchtst_mae `
    trainer=h200 trainer.max_epochs=20
```

## Contributing back to the recipe

If you had to edit `_TELEMETRY_FILENAMES`, `_ANOMALY_FILENAMES`, or the
timestamp / anomaly column lists in `scripts/convert_ops_sat_to_parquet.py`
to match a new OPS-SAT release, commit that edit — future contributors
will thank you.

## When to include OPS-SAT vs skip it

**Include** when:
- You're pretraining a foundation model and need real satellite data
  volume beyond NASA Telemanom
- You want a "cross-mission generalization" story (train on NASA,
  test on ESA)
- Reviewers ask for public real-data validation beyond one source

**Skip** when:
- CPU-only smoke testing (parquet loads fast but training on 100+
  channels of real data on CPU is not fun)
- Iterating on a synthetic recipe (use `presets/` variants instead)
- Your air-gap policy hasn't cleared ESA-hosted data yet

## Troubleshooting

| Symptom | Fix |
|---|---|
| `404` from download URL | Zenodo rotated the record. Update `DATASET_URL` in the download script. |
| `No OPS-SAT file matching …` from converter | Edit `_TELEMETRY_FILENAMES` etc. in the converter to match the actual filenames. |
| `No timestamp column found` from converter | Edit `_TIMESTAMP_COLUMNS` to include your CSV's time column name. |
| Empty labels after conversion | Your release may not include the anomaly file, or column names differ. Check `_ANOMALY_START_COLUMNS` / `_ANOMALY_END_COLUMNS`. |
| Out-of-memory during conversion | Use `--downsample 10` or `--channels <subset>` to reduce size. |
| Loader loads but AUCs are NaN | The val split contains no positives. Use `--channels` to pick a channel with anomalies. |

## Provenance for security review

| Item | Value |
|---|---|
| Data owner | European Space Agency |
| Publisher | ESA + collaborators (Airbus, GMV) |
| Dataset name (as of this writing) | OPS-SAT-AD |
| License | ESA public data (verify at release page) |
| Contains PII? | No — spacecraft housekeeping only |
| Air-gap compatible? | Yes — one-time download, then sneakernet |
| Falls under export control? | Verify with your legal team; ESA satellite ops data is generally cleared for open release |
