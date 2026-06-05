# dgx_ts_lab.datasets.cyber

Phase 8 — cybersecurity-flavored datasets. Treat command sequences, operator activity, and subsystem side-channel patterns as `TelemetryDataset` types so existing detectors + the Fabric loop work unchanged.

## Files

| File | Class | Registry key | Use |
|---|---|---|---|
| `_tokenizer.py` | `CommandTokenizer` | — | Multi-token encoder: `[<CMD>, opcode_id, param1_id, ...]`. Single source of truth shared with Phase 11 procedure synth. |
| `command_sequence.py` | `CommandSequenceDataset` | `command_sequence` | D11. Discrete command-token stream as a 1-channel `TelemetryDataset` (tokens stored as float32 for Protocol, cast back to long in the detector). |
| `activity_window.py` | `ActivityWindowDataset` | `activity_window` | D12. Rolling statistical features (login rate, command rate, command diversity, time-of-day, session length). Aux label `operator_id` per timestep tells `OperatorFingerprintDetector` which baseline to compare against. |
| `side_channel.py` | `SideChannelDataset` | `side_channel` | D13. Generic adapter — wraps ANY existing `TelemetryDataset` + a behavior-summary function, exposes a derived `TelemetryDataset`. Apply behavior AD to EPS / ADCS / anywhere. |

## How they wire into existing detectors

| Dataset | Default detector | Notes |
|---|---|---|
| `command_sequence` | `sequence_transformer` (BERT-MLM) | Token stream → MLM perplexity per step |
| `activity_window` | `operator_fingerprint` (Mahalanobis) | Activity features → per-operator embedding distance |
| `side_channel` | Any neural detector (rolling_mean, patchtst_mae, sat_tsfm) | Derived features are continuous, fit existing pipeline |

## Tokenizer vocab structure

```
[0..3]                       special tokens: <PAD> <CMD> <UNK> <MASK>
[4..4+N_opcodes-1]           opcode tokens
[4+N_opcodes..]              parameter tokens
```

Special tokens (`PAD_TOKEN`, `CMD_TOKEN`, `UNK_TOKEN`, `MASK_TOKEN`) are skipped by the MLM masking logic so we don't waste capacity predicting structural tokens.

## Synth generators

See [`../synthetic/cyber/`](../synthetic/cyber/) for realistic command-stream and operator-activity generators with labeled injection patterns.

## See also

- Behavior-style detectors: [`packages/dgx_ts_lab/src/dgx_ts_lab/models/behavior/README.md`](../../models/behavior/README.md)
- Sequence transformer: [`packages/dgx_ts_lab/src/dgx_ts_lab/models/from_scratch/sequence_transformer.py`](../../models/from_scratch/sequence_transformer.py)
- Phase 8 plan: [`docs/phase_plans/phases_6_through_11.md`](../../../../../../docs/phase_plans/phases_6_through_11.md)
