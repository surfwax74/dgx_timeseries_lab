# `dgx_ts_lab.datasets.multimodal`

Phase 10 multi-modal subpackage. Bundles **three aligned modalities** as a
single `TelemetryDataset`:

| Modality   | Source              | Channels | Per-step encoding                          |
|------------|---------------------|----------|--------------------------------------------|
| Telemetry  | LayeredSynth / NASA | C_tel    | continuous float32                         |
| Commands   | sparse events       | 3        | `[count, last_opcode_id, last_param_id]`   |
| Logs       | sparse events       | 3        | `[count, max_severity, last_source_id]`    |

All three are resampled / bucketed onto a **common 1-Hz time grid** and
concatenated channel-wise into the standard `TelemetryWindow.tensor`. The
window's `provenance` carries `channel_modalities` so models can split the
tensor back into its three modality views.

## Files

| File                            | Purpose                                                                              |
|---------------------------------|--------------------------------------------------------------------------------------|
| `_log_tokenizer.py`             | `LogSeverity` IntEnum + `LogTokenizer` (small structured-event vocab — no free text) |
| `event_bucketer.py`             | `CommandEventBucketer` + `LogEventBucketer` reduce bursty timestamped events to (n_bins, 3) |
| `multimodal_dataset.py`         | `MultiModalDataset` — concatenates the three modalities, implements `TelemetryDataset` Protocol |
| `synth_multimodal_leo.py`       | `generate_multimodal_leo()` — correlated synthetic LEO telemetry+commands+logs (fault-coincident events) |
| `__init__.py`                   | Public surface + side-effect dataset registry registration                            |

## Channel layout (locked)

`MultiModalDataset` concatenates in this fixed order:

```
[ telemetry_0 ... telemetry_{C_tel-1} | cmd_count cmd_opcode cmd_param | log_count log_severity log_source ]
```

That order is also what `SatMultiModalDetector` expects when splitting
`batch["x"]` into the three streams. Don't reorder these without updating
`channel_modalities` propagation everywhere.

## Cross-modal correlation (the reason this exists)

`generate_multimodal_leo` injects three classes of correlation:

1. **Routine cadence**: ~30 cmds/hr, 1-Hz heartbeat INFO logs.
2. **Mode transitions**: ModeMachine flips → telemetry shifts + INFO log.
3. **Fault-coincident events**: telemetry fault → ERROR/FATAL log at the
   same timestamp; a fraction (`fault_correlated_priv_rate`) get a
   priv-escalation command (`FW_UPLOAD`) dropped just BEFORE the fault.

The cross-modal MAE pretraining on `SatMultiModalDetector` exploits all
three: masking telemetry patches forces the model to predict the missing
signal from co-occurring commands and logs.

## Usage

```python
from dgx_ts_lab.datasets.multimodal.synth_multimodal_leo import generate_multimodal_leo
from dgx_ts_lab.datasets.multimodal.multimodal_dataset import MultiModalDataset

kwargs = generate_multimodal_leo(n_seconds=3600, seed=0, n_telemetry_channels=6)
ds = MultiModalDataset(**kwargs)
for win in ds.windows(length=256, stride=256):
    parts = ds.split_window_by_modality(win)
    # parts["telemetry"]: (256, 6); parts["commands"]: (256, 3); parts["logs"]: (256, 3)
```

Or via the registry (used by Hydra configs):

```python
from dgx_ts_core.registry import DATASET_REGISTRY
ds = DATASET_REGISTRY.create("synth_multimodal_leo", n_seconds=3600, n_telemetry_channels=6)
```

Hydra config: `configs/dataset/multimodal_synth.yaml`. Smoke experiment:
`configs/experiment/phase10_multimodal.yaml`.

## Extending

- **New event modality**: add a `*EventBucketer` to `event_bucketer.py`
  that produces a `(n_bins, F)` float32 array, then extend
  `MultiModalDataset` to accept + concatenate it. Update
  `channel_modalities` and `SatMultiModalDetector._split_modalities`.
- **New synthetic generator**: write a `generate_multimodal_X.py` next to
  `synth_multimodal_leo.py`, return the same kwargs dict, and register it
  with `@DATASET_REGISTRY.register("X")`.
- **Replace synthetic with real**: write a loader that yields the same
  `(telemetry, commands, logs)` triple aligned to a common rate, then
  hand them to `MultiModalDataset(...)`. The model contract is unchanged.

## Tests

`packages/dgx_ts_lab/tests/test_phase10_multimodal.py` covers tokenizer,
bucketers, dataset shape / split / window, registry registration, and
model end-to-end (fit → loss → score → save/load).
