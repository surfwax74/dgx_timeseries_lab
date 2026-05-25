# Foundation model provisioning (air-gap)

Phase 3 foundation models (Chronos, MOMENT, Moirai) need pretrained weights. They are 100–500 MB each and must be sneakernetted onto the DGX — never auto-downloaded at runtime.

There are two valid locations the loader checks (in order):

1. **MLflow Model Registry** (primary, production) — URIs like `models:/chronos-t5-small/Production`.
2. **Local filesystem fallback** — `data/models/<org>/<model_name>/`.

You can use either or both. The dev/Windows pattern is usually local-only; the DGX uses both.

## On a connected machine — fetch weights

```bash
# Pick one — repeat per model
huggingface-cli download amazon/chronos-t5-tiny --local-dir data/models/amazon/chronos-t5-tiny
huggingface-cli download AutonLab/MOMENT-1-small --local-dir data/models/AutonLab/MOMENT-1-small
huggingface-cli download Salesforce/moirai-1.0-R-small --local-dir data/models/Salesforce/moirai-1.0-R-small
```

Each `<org>/<model>/` directory should contain at minimum `config.json` and the weight files (`model.safetensors` or `pytorch_model.bin`). Tokenizer files are also typically included.

## Sneakernet to the air-gapped DGX

```bash
# On the source machine:
tar czf foundation_models.tar.gz data/models/
# Transfer the tarball (USB, IPNet, etc.)

# On the DGX:
tar xzf foundation_models.tar.gz -C /path/to/dgx_timeseries_lab/
```

Verify:

```bash
ls data/models/amazon/chronos-t5-tiny/
# Should show: config.json, generation_config.json, model.safetensors, tokenizer.json, …
```

## Path A: filesystem-only (simpler — start here)

Once the weights are at `data/models/<org>/<model_name>/`, the loader resolves them automatically.

```powershell
uv run dgx-ts train model=chronos_zero dataset=parquet trainer=single_cpu mode=zeroshot
```

`configs/model/chronos_zero.yaml` references `model: amazon/chronos-t5-tiny` — the loader looks for `data/models/amazon/chronos-t5-tiny/`. Done.

## Path B: MLflow Registry (production)

### 1. Start the Registry server

```bash
./scripts/setup_mlflow_registry.sh
```

This launches an MLflow server on `http://127.0.0.1:5000` with:
- SQLite backend at `mlflow_registry/registry.db`
- Local artifact store at `mlflow_registry/artifacts/`

Air-gap safe — nothing leaves the box. Run it under systemd / a tmux session so it persists.

### 2. Register the foundation models

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run python scripts/register_foundation_models.py
```

The script walks `data/models/<org>/<model>/`, uploads each snapshot to the artifact store, and creates a registered model named `<model>` at the `Production` stage. Repeat after adding new weights or upgrading versions.

### 3. Configs can now use Registry URIs

```yaml
# configs/model/chronos_zero.yaml
_target_key: chronos
model: "models:/chronos-t5-tiny/Production"
```

Runtime: the loader sees the `models:/...` prefix, asks MLflow to download the artifact, returns a local path to the model dir. Cached after first fetch.

## Dual-path behavior

The loader is implemented in [`packages/dgx_ts_lab/src/dgx_ts_lab/models/foundation/_loader.py`](../packages/dgx_ts_lab/src/dgx_ts_lab/models/foundation/_loader.py). The resolution order is:

1. If the string looks like a path and exists → use it directly.
2. If the string starts with `models:/` → try MLflow Registry. On any error (server unreachable, no such model), fall back to `data/models/...`.
3. Otherwise → look up `data/models/<string>/`.

This makes dev on Windows (no Registry running) and prod on the DGX (Registry available) use the same configs unchanged.

## Verifying a model is reachable

```powershell
uv run python -c "from dgx_ts_lab.models.foundation._loader import resolve_model_path; print(resolve_model_path('amazon/chronos-t5-tiny'))"
```

Prints the resolved local directory, or raises `FileNotFoundError` with provisioning instructions.

## What if a model isn't available?

`dgx-ts train model=chronos_zero ...` will run with a small **untrained** T5 stub (defined in `_ChronosModule.__init__`'s except branch). Training still works; predictions are just random until you provide real weights. This makes development without weights possible.

For tests, the same fallback path is what makes the unit tests run on any machine.
