# Forecasting + RUL Bake-off — scoping doc

**Status:** W1 in progress (scoping locked, metrics modules building).
Five-task work package, ~1 week of engineering. See task list `#92`–`#96`.

A second bake-off, **separate from the AD bake-off**, that benchmarks
forecasters and RUL estimators on satellite telemetry. This is where
Prophet, ETS, and the prognostics use cases belong — and where our
existing Sat-TSFM multi-task `RULRegressorHead` finally gets a leaderboard
slot.

## Data strategy: real vs synthetic (the hard split)

**Forecasting** and **RUL** have fundamentally different data requirements,
which drives our real-vs-synthetic split:

| Task | Real data OK? | Why |
|---|---|---|
| Forecasting | ✅ YES | Needs continuous long-runway signal. OPS-SAT (~4 months) and NASA Telemanom (~weeks/channel) both work. |
| RUL         | ❌ NO   | Needs run-to-failure trajectories with EOL labels. Neither NASA nor OPS-SAT provide these — spacecraft are still flying. |

**Real datasets in the forecasting side** (multi-step-ahead evaluation):
- **OPS-SAT-AD** (~4 months, ~150 channels @ 1 Hz) — ample runway for
  h=64 through h=1024 evaluations, mission-realistic seasonality.
- **NASA SMAP/MSL** (~weeks per channel) — enough for h=64 through h=256;
  serves as the "small but real" baseline against OPS-SAT.

**Synthetic-only for RUL** (because you need EOL labels the real
missions can't give you):
- Scenario A — `battery_soc_degradation` (6 months synth, EOL = SoC <30%)
- Scenario B — `fuel_mass_projection` (2 years synth, EOL = fuel <reserve)

If NASA C-MAPSS or similar public RUL-labeled data is later added,
it slots in beside these two scenarios. Do NOT retrofit RUL onto
NASA-Telemanom or OPS-SAT — the labels aren't there and any RUL
number produced would be meaningless.

---

## Why a separate bake-off

Forecasting and AD answer different questions:

| Task | Question | Right metric | Horizon |
|---|---|---|---|
| AD (current bake-off) | "Is the system anomalous *right now*?" | ROC-AUC, F1, point-adjusted F1 | now |
| **Forecasting** | "What will the value be N steps ahead?" | MAE, RMSE, sMAPE, pinball loss, CRPS | minutes to months |
| **RUL** | "How long until the component fails?" | NASA S-score, early/late MAE, hit rate at threshold | hours to years |

Putting these in the AD bake-off would give misleading numbers — ROC-AUC
isn't defined for a continuous forecast, and MAE isn't meaningful for a
binary AD decision. Different tasks, different leaderboards.

---

## Models included

### Forecasting

| Model | Why it's here | Status |
|---|---|---|
| **Prophet** (Meta) | Long-horizon prognostics, holiday mechanism for burns/eclipse seasons | Add (W3) |
| **ETS / Holt-Winters** (statsmodels) | Classical baseline; no heavyweight deps | Add (W3) |
| **Chronos** (Amazon) | Already in lab; native forecaster | Reuse |
| **MOMENT** (CMU AutonLab) | Already in lab; reconstruction → forecast | Reuse |
| **Moirai** (Salesforce) | Already in lab; native forecaster | Reuse |
| **TimesFM** (Google) | Roadmap Tier 1; designed for forecasting | Add (after foundation roadmap) |
| **TTM** (IBM) | Roadmap Tier 1; fastest LoRA on a 3080 | Add (after foundation roadmap) |
| **PatchTST** (in-house) | We use it for AD but it's natively a forecaster | Reuse with `mode=forecast` |
| **`rolling_mean`** baseline | Persistence model: predict last observed value | Reuse |

### RUL

| Model | Why it's here | Status |
|---|---|---|
| **`rul_regressor` head on Sat-TSFM** | Already exists from Phase 6; finally has a leaderboard slot | Reuse |
| **Battery PINN residual** | Physics-grounded battery degradation | Reuse |
| **Prophet** with linear extrapolation of trend to failure threshold | Simple analytical RUL via Prophet's slope projection | Add (W3 same code path as forecasting) |
| **ETS** with same extrapolation pattern | Classical RUL baseline | Add (W3) |

---

## Demo scenarios (W2)

Two synthetic scenarios that exercise the strengths of different model
families. Both extend `LayeredSyntheticDataset` so they integrate
cleanly with the existing infrastructure.

### Scenario A: `battery_soc_degradation`

- **Duration**: 6 months synthetic, 1-hour aggregated
- **Signal**: battery SoC with orbital cycle + capacity degradation +
  eclipse seasons + occasional load-shedding events
- **Task**: forecast SoC trajectory weeks ahead; predict RUL = days until
  SoC drops below mission-critical threshold (e.g. 30%)
- **Prophet sweet spot**: multi-day seasonality, slow trend, discrete
  "holiday" events. Should perform competitively here.
- **Where the transformers shine**: capturing non-stationary degradation
  rate changes (battery wear-out accelerates) — Prophet's linear-trend
  prior misses these.

### Scenario B: `fuel_mass_projection`

- **Duration**: 2 years synthetic, daily aggregated
- **Signal**: monotonically decreasing fuel mass with discrete burns
  (~weekly station-keeping, occasional collision-avoidance)
- **Task**: forecast fuel level 6 months ahead; predict RUL = days until
  fuel falls below disposal-burn reserve
- **Prophet sweet spot**: holiday mechanism literally maps to discrete
  burns. Strong fit.
- **Where the transformers shine**: anticipating *unscheduled* burn
  events from precursor telemetry (collision-avoidance prediction).

---

## Metrics (W1)

### `evaluation/forecasting_metrics.py`

Point forecasts:

- **MAE** — mean absolute error
- **RMSE** — root mean squared error
- **sMAPE** — symmetric MAPE (the standard one for TS competitions)
- **MASE** — mean absolute scaled error (vs. naive seasonal forecast)

Probabilistic forecasts:

- **Pinball loss** at quantiles `[0.1, 0.5, 0.9]`
- **CRPS** — continuous ranked probability score
- **Coverage** — fraction of true values inside the predicted 80% / 95%
  interval (calibration check)

### `evaluation/rul_metrics.py`

- **NASA S-score** — asymmetric penalty: late predictions cost more than
  early ones (the standard CMAPSS / NASA RUL metric)
- **MAE on RUL** — straight absolute error
- **Early/Late hit rate** — fraction of predictions within ±N days of
  true failure
- **Calibration plot data** — predicted RUL vs. true RUL across the test
  set

---

## Reusable infrastructure

Most of the bake-off plumbing already exists. Specifically:

| Component | Reuse from | Adapt? |
|---|---|---|
| Benchmark orchestrator | `evaluation/benchmark.py` | Yes (W4 — write `forecasting_benchmark.py` mirroring the AD one) |
| Per-run npz output | Same `dgx-ts benchmark` pattern | Yes (different array shapes) |
| Visualization helpers | `evaluation/visualize.py` | Add 3 new plot functions (W4) |
| CLI | `cli/main.py` dispatcher | Add `forecast-bench` subcommand (W4) |
| Trainer | `LightningTrainer` | Yes — neural forecasters fit the same Fabric loop |
| MLflow logging | `tracking/mlflow_logger.py` | Yes (different metric names) |

---

## CLI sketch (W4)

```bash
# Run the forecasting+RUL bake-off
dgx-ts forecast-bench experiment=forecasting_rul_bakeoff

# Output: benchmark_reports/forecasting_rul_bakeoff/
#   benchmark_report.md          ranked by MASE (forecasting) + NASA S-score (RUL)
#   benchmark_report.json
#   <model>__<scenario>__h<horizon>__s<seed>.npz   (forecast, actual, intervals)
#   figures/
#     forecast__<scenario>.png   forecast lines + confidence intervals + actual
#     error_vs_horizon.png       per-model error growth with horizon length
#     rul_calibration.png        predicted RUL vs true RUL, scatter
```

---

## Demo angle for procurement

Three slides:

1. **"Forecasting and AD are different tasks"** — table contrasting
   metrics, horizons, model families. Pre-empts the "why isn't Prophet
   in the AD slide" question.
2. **"Prognostics leaderboard"** — bar chart of NASA S-score per model
   on battery degradation. Prophet competitive on slow trend; Sat-TSFM
   wins on capturing degradation-rate changes.
3. **"One foundation, two tasks"** — the same Sat-TSFM backbone with
   different heads delivers both AD (Phase 6) and RUL prognosis (this
   bake-off). The procurement story: "buy one model, get two ops
   capabilities."

The third slide is the strongest argument for the multi-task design —
foundation models earn their cost when they amortize across tasks.

---

## Out of scope (for v1)

- **Probabilistic neural forecasters** (DeepAR-style quantile output) —
  worth a follow-up; current Chronos / Moirai / MOMENT adapters give
  point forecasts.
- **Cross-channel forecasting** (forecast bus voltage *given* current
  trajectory) — Phase 10 multi-modal pretrains for this but the
  forecasting evaluation here stays per-channel for simplicity.
- **Online RUL updates** (RUL prediction that updates each step rather
  than per-window) — useful for ops dashboards; not needed for the
  bake-off comparison.

These are roadmap candidates after v1 ships.

---

## Sequencing summary

| Task | Owner | Days | Output |
|---|---|---|---|
| W1: scoping + metrics modules | TBD | 1.5 | This doc, forecasting_metrics.py, rul_metrics.py |
| W2: dataset wrapper + 2 scenarios | TBD | 1.5 | horizon_dataset.py, battery + fuel scenarios |
| W3: Prophet + ETS adapters | TBD | 1.0 | 2 baseline detectors, 2 YAML configs |
| W4: orchestrator + viz + CLI | TBD | 1.5 | forecasting_benchmark.py, 3 plot helpers, forecast-bench CLI |
| W5: tests + docs | TBD | 1.0 | Tests, cookbook entry, README update |
| **Total** | | **~1 week** | Procurement-ready forecasting+RUL story |

---

## Cross-references

- AD bake-off (the current one): [`experiments_cookbook.md`](experiments_cookbook.md)
- Algorithm inventory: [`../packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md)
- Foundation model roadmap (TimesFM / TTM / TimeMoE are also forecasters): [`foundation_model_roadmap.md`](foundation_model_roadmap.md)
- Multi-task heads (where `RULRegressorHead` lives today): [`../packages/dgx_ts_lab/src/dgx_ts_lab/models/heads/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/models/heads/README.md)
