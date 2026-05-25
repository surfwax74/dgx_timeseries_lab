# configs/dataset/presets/

Canonical pre-built dataset configurations. Use these as starting points or for reproducible benchmarks.

## Files

| YAML | Subsystem | Channels | Use case |
|---|---|---:|---|
| `leo_eps_24h.yaml` | EPS (minimal) | 6 | Phase 1 smoke test. Fast to generate; ideal for debugging. |
| `leo_eps_full_24h.yaml` | EPS (comprehensive, redundant) | **83** | Realistic benchmark target. Full Side-A/Side-B redundancy: 2 solar arrays, 2 batteries, 2 PDUs, 2 PCUs, 5 major loads. Use for the actual bake-off. |

## leo_eps_full_24h.yaml — channel layout

```
Solar Arrays (SA+X, SA-X)        22 channels
  per wing: 3 panel temps, total V + I, SADA angle + motor I, 4 cell-string currents

Batteries (BAT_A, BAT_B)         30 channels
  per battery: terminal V, current, 4 cell-bank temps, SoC, 8 cell voltages

PDUs (PDU_A, PDU_B)              18 channels
  per PDU: pri/sec bus V+I, 4 representative load currents, internal temp

PCUs (PCU_A, PCU_B)               8 channels
  per PCU: regulator V+I, MPPT state, internal temp

Major loads                       5 channels
  payload, comms TX, 2 heaters, OBC
```

### Cross-channel realism (L3)

- Each panel current is a `SumCoupling` over its four cell-string currents.
- Each PDU primary-bus current is a `SumCoupling` over its representative loads + the secondary-bus draw.
- Bus voltage sags slightly under bus current via `InverseCoupling`.
- Payload power feeds into PDU_A load 3; comms TX feeds into PDU_B load 3.
- Cell voltages within a battery share common-mode noise via `CorrelatedGaussianNoise` over an 8×8 covariance.

### Fault types injected (L6)

| Fault | Where | Pattern |
|---|---|---|
| Point transient | bus voltages | random impulses |
| Telemetry dropout | cell voltages, cell-string currents | brief zero-fills |
| SADA encoder stuck | SADA angles | hold-at-onset for 1–5 min |
| Sensor cal drift | battery temps | bias ramp + persist |
| Cell imbalance | one cell each battery | persistent negative drift |
| Bus undervoltage osc | PDU_A pri V | 90 s of 0.3 Hz oscillation |
| Correlation break | one BAT_A cell | additive divergence from peers |
| Cross-strap switch | PDU_A load 4 | step change mid-run |
| Mode confusion | payload power | high draw during eclipse |

## How to invoke

```powershell
dgx-ts train experiment=phase1_layered           # 6-channel smoke (fast)
dgx-ts synth dataset=presets/leo_eps_full_24h    # write 83-ch dataset to parquet
dgx-ts train dataset=parquet model=rolling_mean  # load parquet, train
```

## Adding a new preset

Presets live here when they're (1) realistic enough to be a benchmark and (2) used by more than one experiment. One-off configs belong in `configs/experiment/`.

Recommended naming: `<orbit_or_mission>_<subsystem>_<duration>.yaml` — e.g., `geo_comms_7d.yaml`, `mars_rover_tcs_48h.yaml`.

## See also

- Parent: [`../README.md`](../README.md)
- Generator components: [`packages/dgx_ts_lab/src/dgx_ts_lab/datasets/synthetic/layered/README.md`](../../../packages/dgx_ts_lab/src/dgx_ts_lab/datasets/synthetic/layered/README.md)
