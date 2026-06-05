# dgx_ts_core.data

Data-side contracts: the window type that crosses dataset ↔ model, plus the dataset Protocol every loader implements.

## Files

| File | What it defines |
|---|---|
| `window.py` | `TelemetryWindow` — the only data type crossing the dataset/model boundary. Immutable, numpy-backed. |
| `schema.py` | `Channel`, `Subsystem`, `Units` — per-channel metadata enums. |
| `dataset.py` | `TelemetryDataset` Protocol + `DatasetStats` dataclass. |
| `splits.py` | `SplitScheme`, `SplitStrategy` — train/val/test partitioning rules. |
| `__init__.py` | Re-exports everything for `from dgx_ts_core.data import …`. |

## Invariants

- All array fields are numpy (never torch) so this module stays framework-free.
- `TelemetryWindow.__post_init__` validates shapes at boundaries; internal callers can construct freely.
- All dataclasses are `frozen=True, slots=True`.

## See also

- Parent: [`../README.md`](../README.md) (dgx_ts_core overview)
- Sibling: [`../models/README.md`](../models/README.md) (consumes `TelemetryWindow`)
