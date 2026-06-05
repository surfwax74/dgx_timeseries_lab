# dgx_ts_lab.datasets.synthetic.cyber

Phase 8 synthetic generators for cybersecurity scenarios. Each produces a ready-to-use `TelemetryDataset` instance (or a kwargs dict that can be passed into the matching dataset constructor).

## Files

| File | Generator | Output dataset | Registry key |
|---|---|---|---|
| `command_sequence_gen.py` | `generate_command_sequence` | `CommandSequenceDataset` | `synth_command_sequence` |
| `operator_traffic_gen.py` | `generate_operator_traffic` | `ActivityWindowDataset` | `synth_operator_traffic` |

## Command-sequence injection patterns

Five labeled injection types coexist in one stream (each gets a class id in `aux_labels.injection_type`):

| Class id | Pattern | Description |
|---|---|---|
| 0 | (no injection) | Normal traffic — routine + scheduled + operator commands |
| 1 | `priv_escalation` | Operator issues a command class only admins should issue |
| 2 | `flooding` | Same command repeated 10–30× in a row |
| 3 | `replay` | A 5-command subsequence verbatim repeated 50 commands later |
| 4 | `sequence_anomaly` | Syntactically valid but operationally meaningless ordered sequence |

Adjust the per-class injection rates via the YAML knobs in `configs/dataset/cyber/cmdseq_synth.yaml`.

## Operator-activity personas

Three default personas (`alice`, `bob`, `carol`) with distinct fingerprints:

| | alice | bob | carol |
|---|---|---|---|
| Active hour center | 09:00 | 14:00 | 22:00 |
| Command rate (cmds/min, lognormal mean) | ~2.2 | ~1.5 | ~3.0 |
| Diversity (Shannon nats) | 1.8 | 1.2 | 2.2 |
| Session length (min, lognormal mean) | ~20 | ~55 | ~12 |

Impersonation = label True when the claimed `operator_id` differs from the actual generating persona. Adjust frequency via `impersonation_rate`.

## Air-gap

Pure synthetic — no external data dependencies. Deterministic from `(config, seed)`.

## See also

- Cyber datasets: [`../../cyber/README.md`](../../cyber/README.md)
- Phase 8 bake-off: [`configs/experiment/phase8_cyber.yaml`](../../../../../../../configs/experiment/phase8_cyber.yaml)
