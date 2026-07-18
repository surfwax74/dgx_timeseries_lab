# Pretraining Corpus Roadmap

**Goal.** Give the DGX pretraining runs enough diverse, mission-realistic
telemetry that a single Sat-TSFM checkpoint transfers to any real
spacecraft. Today the corpus is 6 EPS variants (~7 days of continuous
LEO data). The 30-mission plan below scales it to enough breadth that
zero-shot generalization to unseen missions is a defensible claim.

## Current state (Phase A — shipping now)

Six EPS variants, sharing a 6-channel schema
(`bus_voltage, bus_current, battery_soc, panel_temp_x, panel_temp_y,
payload_power`), each 24 h at 1 Hz. Materialize with
`pwsh scripts/build_corpus.ps1` (Windows) or `bash scripts/build_corpus.sh`
(Linux/DGX), then train with `dgx-ts train experiment=dgx_pretrain_corpus`.

| Member            | Regime                                    | Seed |
|-------------------|-------------------------------------------|------|
| `leo_eps_24h`     | Base / reference                          | 42   |
| `leo_eps_v1`      | Quiet — low fault, subdued noise          | 1001 |
| `leo_eps_v2`      | Stormy — high noise, high fault rate      | 1002 |
| `leo_eps_v3`      | Sun-sync — longer orbital period          | 1003 |
| `leo_eps_v4`      | Aging — LinearDrift + ExponentialAging    | 1004 |
| `leo_eps_v5`      | Payload-heavy — high load / oscillations  | 1005 |

Not in the current 6-channel corpus:
* `leo_eps_full_24h` — 83-channel variant. Kept out of `leo_eps_corpus`
  because the corpus loader requires homogeneous channel schemas across
  members. Used separately for width-scaling experiments.
* `ops_sat` — real ESA data. Different subsystem mix; will land in a
  separate `sat_eps_realdata_corpus` once we have >1 real mission.

**Infrastructure delivered in Phase A**
* `packages/dgx_ts_lab/src/dgx_ts_lab/datasets/parquet_corpus.py` —
  `ParquetTelemetryCorpus` (registered as `parquet_telemetry_corpus`).
* `configs/dataset/cached/leo_eps_corpus.yaml` — union alias.
* `configs/experiment/dgx_pretrain_corpus.yaml` — H200 FSDP experiment.
* `scripts/build_corpus.{ps1,sh}` — batch materializer, `-DryRun`/`-Only`/
  `-Force` flags.
* `packages/dgx_ts_lab/tests/test_parquet_corpus.py` — 7 unit tests.

## Phase B — Multi-subsystem breadth (2–4 weeks)

Add corpora for the other subsystems so Sat-TSFM is not just an "EPS
model". Each is a 6-mission-variant sweep parallel to Phase A, following
the same pattern.

| Subsystem | Corpus name              | Channel families                          |
|-----------|--------------------------|-------------------------------------------|
| ADCS      | `leo_adcs_corpus`        | quaternions, wheel speeds, gyro rates     |
| TCS       | `leo_tcs_corpus`         | 8-zone panel temps, radiator flow         |
| Comm      | `leo_comm_corpus`        | RF gain, packet queue depths, TWTA power  |
| Propulsion| `leo_prop_corpus`        | tank pressures, valve states, thrust      |
| Payload   | `leo_payload_corpus`     | detector counts, cooler temps, gain modes |

**Effort per subsystem**: ~2 days each (base preset + 5 variants + cached
aliases + experiment YAML + doc entry). The corpus dataset code
(`ParquetTelemetryCorpus`) is fully generic — no new code required.

**Deliverable at the end of Phase B**: `dgx_pretrain_multisubsystem.yaml`
that unions all 5 subsystem corpora into one meta-corpus, with a
per-subsystem `SubsystemMoE` head routing during finetuning.

## Phase C — Real-mission augmentation (4–8 weeks, gated on data access)

Fold in as many public real-mission datasets as licensing allows.
Candidate list (in order of accessibility):

| Mission        | Source       | Subsystems covered              | Notes                        |
|----------------|--------------|---------------------------------|------------------------------|
| OPS-SAT        | Zenodo/ESA   | mixed subsystems                | Already integrated (#105)    |
| SMAP           | NASA JPL     | EPS + ADCS proxies              | Weak (already using)         |
| MSL rover      | NASA JPL     | Rover telemetry                 | Different regime — non-orbital |
| Sentinel-2 L0  | Copernicus   | Instrument telemetry            | Requires ESA agreement       |
| ISS Columbus   | ESA          | Life support + EPS              | Restricted; separate proposal|
| CryoSat-2      | ESA          | Altimeter + EPS                 | Restricted                   |
| CubeSat swarm  | AWS / MIT-LL | Small-sat EPS+ADCS              | Public via SNSF              |
| Space Domain Awareness | USSF | RF pattern-of-life              | Classified; requires cleared box |

**Each real-mission add is a 4-step protocol**:
1. Download with a `scripts/download_<mission>.py` following the OPS-SAT
   template.
2. Convert with a `scripts/convert_<mission>_to_parquet.py`.
3. Add `configs/dataset/cached/<mission>.yaml` (single-member alias).
4. Add to the relevant subsystem corpus's manifest.

## Phase D — Fault-injection layer (parallel, ongoing)

Each corpus member above can be re-materialized with an aggressive fault
layer for supervised finetuning. This lives in
`configs/dataset/presets/faulted/*.yaml`. The corpus loader treats a
faulted parquet dir identically to a clean one — the union just gets
richer.

## Full-scale 30-mission target

If we execute Phases A–C in full, the corpus reaches:

| Phase    | Members | Days of continuous data | Fresh channels seen |
|----------|---------|--------------------------|---------------------|
| A (done) |    6    | ~7                       |   6                 |
| B        |   36    | ~36                      |  60                 |
| C        |   45+   | ~45 + real-time          | 120+                |
| C+D      |   90+   | ~90                      | 120+ (× 2 regimes)  |

That is the "big enough that a Sat-TSFM checkpoint is credible on any
new spacecraft" threshold and the practical justification for the
8×H200 pretraining budget.

## Roadmap TODOs (in priority order)

1. **Now** — Ship Phase A (this doc + the 6-mission corpus). ✅
2. **Next 2 weeks** — Author `leo_adcs_corpus` variants. Reuse Phase A
   pattern verbatim.
3. **Next 4 weeks** — Author `leo_tcs_corpus`, `leo_comm_corpus`.
4. **Next 8 weeks** — Multi-subsystem meta-corpus + first end-to-end
   H200 pretraining bake-off.
5. **Gate before Phase C** — Confirm data-use rights for each candidate
   real-mission dataset. Do NOT ingest anything with unclear licensing
   into the training corpus.
6. **Phase D** — Design the faulted-mode preset library once we have a
   real signal from A + B that Sat-TSFM is generalizing.
