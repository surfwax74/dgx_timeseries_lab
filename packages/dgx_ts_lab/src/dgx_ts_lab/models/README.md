# `dgx_ts_lab.models` — algorithm inventory

Every concrete `AnomalyDetector` implementation in this repo. Importing
this package self-registers each one with
`dgx_ts_core.registry.DETECTOR_REGISTRY`, after which Hydra configs can
instantiate them by key.

**14 detectors + 3 task heads across 6 algorithm families**, all behind the
same Protocol. The point of the lab is to hot-swap among them — this
README is the catalog.

---

## At a glance — algorithm × phase × tier

| Detector | Registry key | Algorithm family | Intro'd | Realistic tier | Multivariate | Per-step scores |
|---|---|---|---|---|:-:|:-:|
| [`RollingMeanDetector`](baseline/) | `rolling_mean` | Classical residual (z-score) | Phase 0 | CPU | ✓ | ✓ |
| [`PatchTSTMAEDetector`](from_scratch/) | `patchtst_mae` | Channel-independent patch transformer + MAE | Phase 2 | 3080 / A5000 / H200 | ✓ | ✓ |
| [`AnomalyTransformerDetector`](from_scratch/) | `anomaly_transformer` | Prior-vs-series association (KL) | Phase 2 | 3080 / A5000 / H200 | ✓ | ✓ |
| [`DCdetectorDetector`](from_scratch/) | `dcdetector` | Dual-attention contrastive | Phase 2 | 3080 / A5000 / H200 | ✓ | ✓ |
| [`ChronosDetector`](foundation/) | `chronos_zero` / `chronos_lora` | T5-based forecasting foundation model | Phase 3 | A5000 / H200 | ✓ (per-channel) | ✓ |
| [`MoiraiDetector`](foundation/) | `moirai_zero` / `moirai_lora` | Patched encoder-only forecasting foundation | Phase 3 | A5000 / H200 | ✓ | ✓ |
| [`MOMENTDetector`](foundation/) | `moment_zero` / `moment_lora` | Patched encoder + recon foundation | Phase 3 | A5000 / H200 | ✓ | ✓ |
| [`TimesFMDetector`](foundation/) | `timesfm` | Decoder-only patched transformer (Google, 500M) | **roadmap → landed** | A5000 / H200 / 8x H200 | ✓ | ✓ |
| [`TTMDetector`](foundation/) | `ttm` | **MLP-Mixer** (IBM, 1-5M) — fast LoRA on 3080 | **roadmap → landed** | 3080 / A5000 / H200 | ✓ | ✓ |
| [`TimeMoEDetector`](foundation/) | `time_moe` | Decoder-only MoE transformer (Maple728, 200M) | **roadmap → landed** | A5000 / H200 / 8x H200 | ✓ | ✓ |
| [`SatTSFMDetector`](from_scratch/) | `sat_tsfm` | Channel-flexible Sat-TSFM (from-scratch foundation) | Phase 4 | A5000 / H200 / 8x H200 | ✓ | ✓ |
| [`SubsystemMoEDetector`](from_scratch/) | `subsystem_moe` | Per-subsystem Mixture-of-Experts | Phase 4 | A5000 / H200 | ✓ | ✓ |
| [`PINNResidualDetector`](physics/) | `pinn_residual` | Physics-residual wrapper (subtract analytical prediction) | Phase 4 | any | ✓ | ✓ |
| [`SatTSFMMultiTaskDetector`](from_scratch/) | `sat_tsfm_multitask` | Sat-TSFM + 4-head multi-task (AD + fault + RUL + mode) | Phase 6 | A5000 / H200 / 8x H200 | ✓ | ✓ |
| [`SequenceTransformerDetector`](from_scratch/) | `sequence_transformer` | BERT-style MLM over discrete command tokens | Phase 8 | 3080 / A5000 | n/a (discrete) | ✓ |
| [`OperatorFingerprintDetector`](behavior/) | `operator_fingerprint` | Mahalanobis distance vs per-operator baseline | Phase 8 | CPU / 3080 | ✓ | per-window |
| [`ThermalPinn`](physics/) | `thermal_pinn_torch` | Trainable thermal PINN (1D conduction) | Phase 9 | A5000 / H200 | ✓ | ✓ |
| [`AdcsPinn`](physics/adcs/) | `adcs_pinn` | Attitude PINN with 3 SO(3) integrator variants | Phase 9 | A5000 / H200 | ✓ | ✓ |
| [`SatMultiModalDetector`](from_scratch/) | `sat_multimodal` | Three-stream (telem + cmd + log) cross-modal MAE | Phase 10 | A5000 / H200 / 8x H200 | ✓ | ✓ |

Plus three **task heads** (not standalone detectors — they bolt onto Sat-TSFM via [`heads/`](heads/)):

| Head | Registry key | Task | Loss |
|---|---|---|---|
| `FaultClassifierHead` | `fault_classifier` | Multi-class fault classification (8 classes) | Cross-entropy |
| `RULRegressorHead` | `rul_regressor` | Remaining-useful-life regression | Survival-style log-MAE |
| `ModePredictorHead` | `mode_predictor` | Next-mode classification (6 modes) | Cross-entropy |

---

## By subdirectory

### [`baseline/`](baseline/README.md) — classical baselines

**`RollingMeanDetector`** — z-score residual against a windowed mean.
Unbeatable on synthetic point-spike datasets; falls over on drift, stuck-at,
or oscillation faults. The **honest baseline** every learned detector must
beat to justify its FLOPs.

### [`from_scratch/`](from_scratch/README.md) — re-implemented neural detectors

All from-paper, no vendored code. All follow the same neural-detector
contract: `detector.module: nn.Module`, `compute_loss(batch)`,
`compute_score_batch(batch) -> (B, T)`. Fabric loop drives them.

- **`PatchTSTMAEDetector`** — Nie et al., *"A Time Series is Worth 64 Words"*
  (ICLR 2023) + MAE adaptation for AD. Channel-independent patching, random
  patch masking, per-step reconstruction error as anomaly score. Best
  general-purpose detector in the bake-off.
- **`AnomalyTransformerDetector`** — Xu et al., *"Anomaly Transformer"*
  (ICLR 2022). Two-stream attention (prior Gaussian + learned series),
  measures association discrepancy via symmetric KL. Strong on
  point-context anomalies, weak on long-duration drift.
- **`DCdetectorDetector`** — Yang et al., *"DCdetector"* (KDD 2023).
  Dual-attention contrastive learning between patch-wise and
  in-patch-wise representations. Contrastive loss, no reconstruction.
- **`SatTSFMDetector`** — our Phase 4 from-scratch satellite-TS foundation
  model. Channel-flexible (handles 1–256 channels), patchified, FSDP-ready.
  Pretrained via MAE, then frozen or LoRA-tuned downstream.
- **`SubsystemMoEDetector`** — Mixture-of-Experts where each expert
  specializes in a subsystem (EPS, TCS, ADCS, etc.). Routing by the
  `Channel.subsystem` metadata — no learned gate needed.
- **`SequenceTransformerDetector`** — BERT-style MLM over discrete command
  tokens. Phase 8 cyber-AD. Per-token MLM loss; high loss on the held-out
  step = command-sequence anomaly (priv escalation, replay, timing).
- **`SatTSFMMultiTaskDetector`** — Sat-TSFM backbone + composable task
  heads. Trains one encoder against four loss terms simultaneously. The
  "foundation model" demo for the DGX procurement slide.
- **`SatMultiModalDetector`** — three-stream cross-modal MAE.
  Per-modality patch embedders, per-modality self-attn, modality-type
  embedding, shared cross-modal stack, per-modality recon heads. Fault
  signals in telemetry, commands, and logs reinforce each other.

### [`foundation/`](foundation/README.md) — pretrained external models

Adapters around HuggingFace-hosted time-series foundation models. Two
modes for each: `zero` (no parameter updates, threshold-only calibration)
and `lora` (LoRA adapters on the encoder layers).

- **`ChronosDetector`** (Amazon, T5-based) — quantize → tokenize → predict
  next value → anomaly = residual against prediction.
- **`MoiraiDetector`** (Salesforce, patched encoder) — universal forecaster;
  same residual pattern.
- **`MOMENTDetector`** (CMU, patched encoder + recon head) — masked
  reconstruction, naturally suited to AD.

### [`behavior/`](behavior/README.md) — non-neural behavior models

- **`OperatorFingerprintDetector`** — per-operator embedding distribution
  fit at training time. Online, an operator's recent activity-window
  feature vector is scored against the Mahalanobis distance from their
  own baseline. Fires when their behavior shifts (account compromise,
  shift-handover gone wrong, etc.).

### [`physics/`](physics/README.md) — physics-informed detectors

Two patterns: residual wrappers around any inner detector (`pinn_residual`)
and standalone trainable PINNs (`thermal_pinn_torch`, `adcs_pinn`).

- **`PINNResidualDetector`** — generic preprocessor. Computes an analytical
  physics prediction per channel, subtracts it from the input, hands the
  residual to an inner neural detector. Composable with any
  `AnomalyDetector`.
- **`OrbitalResidual`** — sun-angle and eclipse predictions for power
  channels.
- **`ThermalResidual`** — first-order thermal model for temperature channels.
- **`BatteryResidual`** — Coulomb counting + Nernst-style voltage prediction.
- **`ThermalPinn`** — Phase 9, hand-rolled differentiable thermal PINN
  (1D conduction with explicit + implicit Euler + Crank-Nicolson). Loss =
  data MSE + physics residual norm.
- **`AdcsPinn`** — Phase 9, attitude PINN with three SO(3) integrator
  variants (explicit Euler, Lie-group log/exp, quaternion product).
  Choose the integrator per stability/accuracy trade-off.

### [`heads/`](heads/README.md) — task heads (not standalone)

- **`FaultClassifierHead`** — over the encoded sequence, predicts one of 8
  fault classes per timestep. Trained alongside AD loss.
- **`RULRegressorHead`** — predicts log-seconds until next fault; uses a
  ceiling sentinel for "no fault visible."
- **`ModePredictorHead`** — predicts the mode (sun / eclipse / safe /
  payload-active / etc.) `horizon_s` seconds in the future.

Heads attach to a backbone that emits `encode_pooled_steps(x) -> (B, T, D)`.
Currently only `SatTSFMDetector` exposes that interface.

---

## Capabilities matrix

`Capabilities` is what `LightningTrainer` branches on (never on
`isinstance`). Sourced from each detector's `.capabilities` property:

| Detector | `requires_pretraining` | `supports_streaming` | `supports_multivariate` | `output_kind` | `supports_peft` | `supports_export_onnx` |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `rolling_mean` | ✗ | ✓ | ✓ | per-step | ✗ | ✓ |
| `patchtst_mae` | ✓ | ✗ | ✓ | per-step | ✓ | ✓ |
| `anomaly_transformer` | ✓ | ✗ | ✓ | per-step | ✓ | ✗ |
| `dcdetector` | ✓ | ✗ | ✓ | per-step | ✓ | ✗ |
| `chronos_zero` / `chronos_lora` | ✗ / ✓ | ✗ | ✓ | per-step | ✓ | ✗ |
| `moirai_zero` / `moirai_lora` | ✗ / ✓ | ✗ | ✓ | per-step | ✓ | ✗ |
| `moment_zero` / `moment_lora` | ✗ / ✓ | ✗ | ✓ | per-step | ✓ | ✗ |
| `sat_tsfm` | ✓ | ✗ | ✓ | per-step | ✓ | ✓ |
| `subsystem_moe` | ✓ | ✗ | ✓ | per-step | ✗ | ✗ |
| `pinn_residual` | inner's | ✓ | ✓ | per-step | inner's | inner's |
| `sat_tsfm_multitask` | ✓ | ✗ | ✓ | per-step | ✓ | ✓ |
| `sequence_transformer` | ✓ | ✗ | n/a | per-step | ✓ | ✗ |
| `operator_fingerprint` | ✓ | ✓ | ✓ | per-window | ✗ | ✗ |
| `thermal_pinn_torch` | ✓ | ✗ | ✓ | per-step | ✓ | ✗ |
| `adcs_pinn` | ✓ | ✗ | ✓ | per-step | ✓ | ✗ |
| `sat_multimodal` | ✓ | ✗ | ✓ | per-step | ✓ | ✗ |

Notes:
- `output_kind=per_step` means one anomaly score per timestep. `per_window`
  means one per window. Affects how `_score_dataset` aggregates.
- `supports_export_onnx=✓` means the detector has a working Phase 5 ONNX
  trace — see `cli/export.py`. The detectors marked ✗ work fine at training
  + scoring time; ONNX support is a follow-up.
- `pinn_residual` inherits its capabilities from whatever inner detector
  it wraps.

---

## Algorithm-family cheat sheet

When to use what:

| If you need… | Reach for |
|---|---|
| The simplest possible baseline that's hard to beat on point spikes | `rolling_mean` |
| A general-purpose deep AD for multivariate telemetry | `patchtst_mae` |
| Strong performance on subtle point-context anomalies | `anomaly_transformer` |
| A "foundation model" demo without pretraining cost | `chronos_zero` (or any `*_zero`) |
| To fit a small model fast on a 3080 | `patchtst_mae` with `d_model=64`, `n_layers=2` |
| To prove the DGX is needed | `sat_tsfm_multitask` (1.5B params, FSDP) |
| To detect operator behavior changes (cyber) | `operator_fingerprint` |
| To detect anomalous command sequences (cyber) | `sequence_transformer` |
| To enforce a physics prior | `pinn_residual` wrapping any inner detector, OR `thermal_pinn_torch` / `adcs_pinn` for standalone PINNs |
| To use telemetry + commands + logs jointly | `sat_multimodal` |
| To train one model that does AD + classification + RUL + mode prediction | `sat_tsfm_multitask` |

---

## How they're wired in

1. **Self-registration**: each detector lives in a subpackage with a
   `__init__.py` that imports the module, triggering the
   `@DETECTOR_REGISTRY.register("key")` decorator on a factory function.
2. **Configuration**: each detector has a YAML in [`configs/model/`](../../../../../configs/model/) with
   `_target_key: <key>` plus its hyperparameters.
3. **Hot-swap**: `dgx-ts train model=<key>` (or `experiment=<name>` which
   composes a model) instantiates via `DETECTOR_REGISTRY.create(key, **kwargs)`.
4. **Routing**: `LightningTrainer.fit()` branches on `detector.capabilities`
   — never on `isinstance` — to pick between the classical and Fabric
   training paths.

---

## Adding a new detector

1. Implement the `AnomalyDetector` Protocol from `dgx_ts_core.models`.
   Minimum surface: `.capabilities`, `.name`, `.fit()`, `.score()`,
   `.save()`, `.load()`. Neural detectors also expose `.module`,
   `.compute_loss()`, `.compute_score_batch()`.
2. Declare honest `Capabilities` — flags get read at runtime to route
   training and evaluation. Lying here breaks the hot-swap contract.
3. Decorate a factory with `@DETECTOR_REGISTRY.register("my_key")`.
4. Add `from . import my_module` to the relevant subpackage `__init__.py`
   so the registration side-effect fires at import time.
5. Add `configs/model/my_key.yaml` with `_target_key: my_key` plus any
   hyperparameters.
6. Write tests under [`packages/dgx_ts_lab/tests/`](../../../../tests/).
7. Add a row to the inventory table at the top of this README and a
   subsection under the corresponding family directory.

Full walkthrough: [`docs/adding_a_model.md`](../../../../../../docs/adding_a_model.md).

---

## Cross-references

- Subpackage detail: each subdirectory's own `README.md` (linked from the
  table at the top).
- Run recipes per detector: [`docs/experiments_cookbook.md`](../../../../../../docs/experiments_cookbook.md).
- Architecture overview: [`docs/architecture.md`](../../../../../../docs/architecture.md).
- ONNX export status + the Phase 5 lift contract:
  [`docs/lift_to_mlops.md`](../../../../../../docs/lift_to_mlops.md).
- Adding new datasets: [`docs/adding_a_dataset.md`](../../../../../../docs/adding_a_dataset.md).
- **Next-phase backlog** — TimesFM, TTM, TimeMoE, Granite-Code, Phi-4,
  Sparse Autoencoders: [`docs/foundation_model_roadmap.md`](../../../../../../docs/foundation_model_roadmap.md).
