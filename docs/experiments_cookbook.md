# Experiments Cookbook

How to run every experiment in this repo, organized by phase. Single
source of truth — the doc you open when you need to remember the
command for "Phase 6 multi-task" or "the RTX 3080 bake-off."

> **Looking for what each detector actually does?**
> See [`packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md)
> — the algorithm inventory. 16 detectors + 3 task heads, each with
> registry key, paper / algorithm family, intended hardware tier, and
> capabilities. The cookbook tells you *how* to run them; the inventory
> tells you *what they are*.

All commands assume:

- repo root is the working directory
- the venv is at `.venv/` (created by `uv sync`)
- on Windows you call `.\.venv\Scripts\python.exe` directly to avoid
  `uv sync` reverting CUDA torch on each `uv run` invocation
- on Linux / DGX you can use either `.venv/bin/python -m dgx_ts_lab.cli.main ...`
  or `uv run --no-sync dgx-ts ...`

For brevity, the recipes below use the short form `dgx-ts <subcommand>`.
Substitute `.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main <subcommand>`
on Windows when not using the `dgx-ts` entry point.

> **Note**: Set `$env:UV_NO_SYNC = '1'` (PowerShell) or
> `export UV_NO_SYNC=1` (bash) once per session if you've previously hit
> the "uv sync reverts CUDA torch" problem.

---

## Quick reference — what to run when

| Goal | Command | Tier | Wall-clock |
|---|---|---|---|
| Smoke test the platform | `dgx-ts train experiment=phase0_smoke` | CPU | ~30 s |
| Inspect synthetic data | `dgx-ts synth dataset=presets/leo_eps_24h` | CPU | ~2 min |
| Side-by-side detector bake-off + figures | `pwsh scripts/quickstart.ps1` or `bash scripts/quickstart.sh` | CPU | ~1 min |
| RTX 3080 production bake-off | `dgx-ts benchmark experiment=rtx3080_bakeoff` | RTX 3080 | ~25 min |
| Foundation models (Chronos / Moirai) | `dgx-ts benchmark experiment=phase3_bakeoff` | A5000 / H200 | ~15 min |
| TSFM smoke (TimesFM / TTM / TimeMoE) | `dgx-ts benchmark experiment=tsfm_smoke` | CPU | ~2 min |
| Load NASA Telemanom channel | `dataset=nasa_smap_channel +dataset.channel=A-1` | CPU | live |
| Load OPS-SAT real ESA data | `dataset=cached/ops_sat` | any | live |
| Materialize 6-mission EPS corpus | `pwsh scripts/build_corpus.ps1` | CPU | ~27 min |
| DGX corpus pretraining | `dgx-ts train experiment=dgx_pretrain_corpus` | 8x H200 | ~6 h |
| FSDP scale demo | `dgx-ts train experiment=phase4_scale` | 8x H200 | ~6 h |
| Multi-task foundation training | `dgx-ts train experiment=phase6_multitask` | A5000 / H200 | ~45 min |
| Explanation report from a run | `dgx-ts explain --run <mlflow-id>` | CPU | ~10 s |
| Cyber AD bake-off | `dgx-ts benchmark experiment=phase8_cyber` | A5000 | ~20 min |
| Thermal + ADCS PINN bake-off | `dgx-ts benchmark experiment=phase9_pinn_bakeoff` | A5000 | ~30 min |
| Cross-modal foundation pretrain | `dgx-ts train experiment=phase10_multimodal` | CPU smoke / DGX prod | 30 s / ~3 h |
| Interactive LLM co-pilot REPL | `dgx-ts copilot --backend mock` | any | live |
| Train a Sparse Autoencoder | `python -m dgx_ts_lab.explanation.sae ...` (library) | CPU / GPU | ~1-40 min |
| Score forecasts (metrics library) | `from dgx_ts_lab.evaluation import forecasting_metrics` | CPU | µs |
| **Full DGX procurement showcase** | `bash scripts/dgx_showcase.sh` | **8x H200** | ~6-8 h |
| Render plots from any benchmark dir | `dgx-ts viz --benchmark-dir benchmark_reports/<name>` | CPU | ~5 s |
| Generate procurement-deck figures | `python scripts/build_capability_cliff.py` | CPU | ~3 s |

---

## Phase 0 — Scaffold smoke

**Proves**: workspace, registries, Lightning + Hydra + MLflow + CLI all wired.

```bash
dgx-ts train experiment=phase0_smoke
```

**Output**: `mlruns/<exp_id>/<run_id>/` with logged params + metrics +
checkpoint, console banner showing val/test metrics on the trivial synth
sine+spike dataset.

**Files**:
- Config: `configs/experiment/phase0_smoke.yaml`
- Detector: `RollingMeanDetector` (classical, no GPU)
- Dataset: `trivial_synth` (sine + spikes, 10k samples)

---

## Phase 1 — Layered synthetic + NASA loaders

**Proves**: composable physics + noise + fault generator; NASA Telemanom
loader; parquet round-trip; layered LEO EPS preset.

### Run it

```bash
# Smoke test the layered synth
dgx-ts train experiment=phase1_layered

# Generate the LEO EPS 24h synthetic dataset to disk (parquet)
# Note: synth takes `dataset=<name>`, not `experiment=`. The presets/ prefix
# is required because the YAML lives at configs/dataset/presets/.
dgx-ts synth dataset=presets/leo_eps_24h

# Same but the full 83-channel EPS preset (used by Phase 2 bake-off)
dgx-ts synth dataset=presets/leo_eps_full_24h
```

### The three dataset patterns — which selector to use

Three parallel dataset patterns depending on what stage of iteration
you're in:

| Selector | What it does | When to use |
|---|---|---|
| `dataset=presets/leo_eps_24h` | **Regenerates in-memory** every call from the recipe (`_target_key: layered_synth`) | Iterating on the preset recipe itself — tweaking channels, faults, noise |
| `dataset=cached/leo_eps_24h` | **Reads from `data/synth/leo_eps_24h/`** on disk (`_target_key: parquet_telemetry`) | Running experiments — instant load, byte-identical across runs |
| `dataset=parquet_telemetry data_path=…` | Same as `cached/` but with a manual path | Loading a parquet directory that isn't a named preset |

The smart helper handles the "materialize once, use forever" flow:

```powershell
# Windows — builds only if data/synth/<name>/ is missing
pwsh scripts/build_dataset.ps1 leo_eps_24h
pwsh scripts/build_dataset.ps1 leo_eps_full_24h
pwsh scripts/build_dataset.ps1 leo_eps_24h -Force        # rebuild anyway
```

```bash
# Linux/DGX
bash scripts/build_dataset.sh leo_eps_24h
bash scripts/build_dataset.sh leo_eps_full_24h --force
```

After the one-time build, every `dgx-ts train / benchmark` command that
references `dataset=cached/<name>` reads the parquet directly. See
[`configs/dataset/cached/README.md`](../configs/dataset/cached/README.md)
for the pattern + staleness caveat.

### Batch-materializing a corpus (Phase A pretraining data)

The 6 EPS mission variants that ship as the DGX pretraining corpus can
be built in one shot:

```powershell
# Windows — builds all 7 members (base + 5 variants + full 83-ch).
pwsh scripts/build_corpus.ps1                                # ~27 min from scratch
pwsh scripts/build_corpus.ps1 -DryRun                        # list only
pwsh scripts/build_corpus.ps1 -Only leo_eps_v1,leo_eps_v3    # subset
pwsh scripts/build_corpus.ps1 -Force                         # rebuild all
```

```bash
# Linux/DGX
bash scripts/build_corpus.sh
bash scripts/build_corpus.sh --dry-run
bash scripts/build_corpus.sh --only leo_eps_v1,leo_eps_v3
```

After the corpus is materialized, train Sat-TSFM on the union with:

```bash
# H200 FSDP full run
dgx-ts train experiment=dgx_pretrain_corpus

# Downgraded workstation dry-run
dgx-ts train experiment=dgx_pretrain_corpus \
       trainer=rtx_3080_single trainer.max_epochs=1
```

The corpus lives at [`configs/dataset/cached/leo_eps_corpus.yaml`](../configs/dataset/cached/leo_eps_corpus.yaml)
and is a `parquet_telemetry_corpus` union of 6 members with a shared
6-channel EPS schema. Roadmap for scaling to 30+ missions lives at
[`docs/pretraining_corpus_roadmap.md`](pretraining_corpus_roadmap.md).

**Output**: `data/synth/<name>/` with `chunk_*.parquet` files (gitignored
by `/data/` rule). The full 83-ch preset is ~200 MB on disk.

**Files**:
- Config groups: `configs/dataset/layered_synth.yaml`, `configs/dataset/presets/*.yaml`
- Components: `packages/dgx_ts_lab/src/dgx_ts_lab/datasets/synthetic/layered/`
- NASA loader: `datasets/nasa_telemanom.py`
- Parquet loader: `datasets/parquet_telemetry.py`
- Corpus loader: `datasets/parquet_corpus.py` (`parquet_telemetry_corpus` registry key)

### The 6 EPS mission-variant presets

The `leo_eps_v1..v5` presets (plus the base `leo_eps_24h`) all share a
6-channel schema so they compose into `cached/leo_eps_corpus.yaml`.

| Preset | Regime | Seed |
|---|---|---|
| `presets/leo_eps_24h` | Base reference | 42 |
| `presets/leo_eps_v1` | Quiet mission (low fault, subdued noise) | 1001 |
| `presets/leo_eps_v2` | Stormy mission (high noise, high fault rate) | 1002 |
| `presets/leo_eps_v3` | Sun-synchronous orbit (6000 s period) | 1003 |
| `presets/leo_eps_v4` | Aging spacecraft (drift + exponential aging) | 1004 |
| `presets/leo_eps_v5` | Payload-heavy (imaging/SAR duty cycle) | 1005 |

Each is 24 h at 1 Hz, ~10 MB on disk after `dgx-ts synth`. Full roadmap
scaling to 30+ missions: [`docs/pretraining_corpus_roadmap.md`](pretraining_corpus_roadmap.md).

---

## Real satellite data (NASA Telemanom + OPS-SAT)

**Proves**: the same trainer / benchmark path that runs on synthetic
data also runs on real spacecraft telemetry. Two public datasets are
wired.

### NASA SMAP / MSL (Telemanom)

Small (~weeks per channel) but easy to fetch. Ships with the loader.

```bash
# Download once (data lands in data/nasa_telemanom/{SMAP,MSL})
bash scripts/download_nasa_telemanom.sh

# Train against a specific SMAP channel
dgx-ts train dataset=nasa_smap_channel +dataset.channel=A-1 \
    model=rolling_mean trainer=cpu

# Or an MSL channel
dgx-ts train dataset=nasa_msl_channel +dataset.channel=T-4 \
    model=patchtst_mae trainer=rtx3080
```

Channels are per-file — the loader emits a single channel at a time.

### OPS-SAT (real ESA data — ~4 months of housekeeping telemetry)

Multi-month runway across ~150 channels. Requires a one-time
connected-machine download, then sneakernet-transferable.

```powershell
# Step 1 — download on a connected machine (~2 GB zip)
python scripts\download_ops_sat.py

# Step 2 — convert to our parquet layout
python scripts\convert_ops_sat_to_parquet.py `
    --raw data\ops_sat_raw --output data\ops_sat

# Step 3 — verify
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main train `
    dataset=cached/ops_sat model=patchtst_mae `
    trainer=cpu trainer.max_epochs=1
```

Full walkthrough incl. sneakernet + URL-rotation troubleshooting:
[`docs/ops_sat_provisioning.md`](ops_sat_provisioning.md).

**When to use each**:

| Dataset | Runway | Use case |
|---|---|---|
| `trivial_synth` | infinite | Smoke test |
| `presets/leo_eps_*` | 24 h × 6 variants | Iterate on recipes |
| `cached/leo_eps_corpus` | ~7 days | DGX pretraining |
| `nasa_smap_channel` / `nasa_msl_channel` | weeks | Real-data forecasting baseline |
| `cached/ops_sat` | ~4 months | Real-data forecasting up to h=1024 |

**Do NOT** compute RUL on NASA or OPS-SAT — neither dataset contains
run-to-failure trajectories. RUL evaluation goes against synthetic
long-runway scenarios only; see the forecasting+RUL section below.

---

## Phase 2 — From-scratch bake-off (PatchTST + AnomalyTransformer + DCdetector)

**Proves**: three transformer-based AD architectures hot-swap behind one
`AnomalyDetector` Protocol; the Lightning Fabric loop drives them all.

### Run it

```bash
# Tiny single-detector smoke (CPU, ~30 s)
dgx-ts train experiment=phase2_smoke

# Original Phase 2 bake-off on the full 83-ch LEO EPS preset
# (CPU-tuned: 1 epoch, wide stride; replace trainer for real numbers)
dgx-ts benchmark experiment=phase2_bakeoff trainer=rtx3080

# Workstation-tuned bake-off (3 detectors, 2 seeds, 20 epochs, bf16)
dgx-ts benchmark experiment=rtx3080_bakeoff

# Quickstart: tiny self-contained 3-detector run + auto-render figures
pwsh scripts/quickstart.ps1            # Windows
bash scripts/quickstart.sh             # Linux/DGX
```

**Output**: `benchmark_reports/<suite_name>/`:
- `benchmark_report.md` — ranked leaderboard
- `benchmark_report.json` — machine-readable
- `<detector>__<dataset>__s<seed>__<split>.npz` — per-run raw (scores,
  labels) so `dgx-ts viz` can rebuild ROC/PR plots later
- `figures/*.png` + `.svg` if you also ran `dgx-ts viz` against it

**Files**:
- Configs: `configs/experiment/phase2_*.yaml`, `configs/experiment/rtx3080_bakeoff.yaml`,
  `configs/experiment/quickstart_viz.yaml`
- Detectors: `models/from_scratch/{patchtst_mae,anomaly_transformer,dcdetector}.py`

---

## Phase 3 — Foundation models (Chronos / MOMENT / Moirai + TimesFM / TTM / TimeMoE)

**Proves**: HuggingFace-hosted foundation models plug into the same
trainer via zero-shot + LoRA fine-tuning paths. Six adapters ship:

| Adapter | Backing | Family | Registry key |
|---|---|---|---|
| Chronos | Amazon | T5 encoder-decoder | `chronos` |
| MOMENT | CMU AutonLab | Reconstruction transformer | `moment` |
| Moirai | Salesforce | Encoder-only forecaster | `moirai` |
| TimesFM | Google | Decoder-only patched transformer (500M) | `timesfm` |
| TTM | IBM | MLP-Mixer (not transformer, 1-5M) | `ttm` |
| TimeMoE | Maple728/Tsinghua | MoE decoder-only (200M) | `time_moe` |

### Run it

```bash
# Original 3-model bake-off across zero-shot and LoRA-finetuned paths
dgx-ts benchmark experiment=phase3_bakeoff trainer=a5000

# Smoke bake-off of the three NEW adapters vs. rolling_mean baseline
# (uses untrained-fallback path — no HF weights needed, ~2 min on CPU)
dgx-ts benchmark experiment=tsfm_smoke

# Just Chronos zero-shot (no training, fast)
dgx-ts train experiment=phase3_bakeoff +mode=zeroshot \
    "suite.detectors=[{key: chronos, params: {model_id: amazon/chronos-bolt-small}}]"
```

**Output**: same shape as Phase 2 — leaderboard + figures.

**Prerequisite**: Foundation model weights either via HF Hub (`huggingface-cli login`)
OR via sneakernet bundle — see `docs/foundation_model_provisioning.md`. Every
adapter falls back to a tiny untrained module if weights are missing, so
tests + smoke bake-offs still run on any box.

**Full model list + versions + companies**: [`docs/foundation_model_roadmap.md`](foundation_model_roadmap.md).

**Files**:
- Configs: `configs/model/{chronos_zero,chronos_lora,moirai_zero,moirai_lora,moment_zero,moment_lora,timesfm_zero,timesfm_lora,ttm_lora,time_moe_lora}.yaml`
- Adapters: `models/foundation/{chronos,moirai,moment,timesfm,ttm,time_moe}.py`
- Smoke config: `configs/experiment/tsfm_smoke.yaml`

---

## Phase 4 — FSDP scale + PINN + Subsystem MoE

**Proves**: Sat-TSFM trains under FSDP on 8x H200; PINN residual wrapper
combines physics with any inner detector; subsystem MoE routes by
channel metadata.

### Run it

```bash
# Sat-TSFM medium (~400M params) under FSDP — DGX target
dgx-ts train experiment=phase4_scale                       # default trainer=h200_fsdp_8x
dgx-ts train experiment=phase4_scale trainer=h200          # single-H200 sanity check
dgx-ts train experiment=phase4_scale trainer=rtx3080 \     # tiny laptop smoke
    model=sat_tsfm_tiny trainer.max_epochs=2

# Physics-informed bake-off (battery / orbital / thermal residuals)
dgx-ts train experiment=phase4_pinn

# Subsystem MoE (per-subsystem expert routing)
dgx-ts train experiment=phase4_moe
```

**Output**: MLflow logs + FSDP-sharded checkpoints in `checkpoints/`.

**Files**:
- Configs: `configs/experiment/phase4_{scale,pinn,moe}.yaml`
- Model variants: `configs/model/sat_tsfm_{tiny,small,medium,large,xlarge}.yaml`
- Trainer variants: `configs/trainer/{rtx3080,a5000,h200,h200_fsdp_8x}.yaml`
- Code: `models/from_scratch/{sat_tsfm,subsystem_moe}.py`, `models/physics/`

---

## Phase 5 — MLOps lift (ONNX + model_card + feature_schema)

**Proves**: any detector with `Capabilities.supports_export_onnx=True`
can emit three artifacts that downstream `mm_mlops` consumes without
importing this repo.

### Run it

```bash
# After a training run produced a checkpoint, export with explicit
# model + dataset (the export CLI doesn't use the experiment= selector;
# `+checkpoint=` is required because it's not in the base config).
dgx-ts export \
    model=patchtst_mae \
    dataset=trivial_synth \
    +checkpoint=checkpoints/patchtst_mae/last.ckpt \
    output_dir=runs/exported_demo

# With Triton ensemble layout (for PINN-wrapped or multi-output models):
dgx-ts export \
    model=patchtst_mae \
    dataset=trivial_synth \
    +checkpoint=checkpoints/patchtst_mae/last.ckpt \
    output_dir=runs/exported_demo \
    +write_triton=true
```

**Output**: `runs/exported_demo/`:
- `model.onnx` (and `model_threshold_baked.onnx` if the detector supports it)
- `model_card.yaml` — metrics, capabilities, threshold, intended subsystem
- `feature_schema.yaml` — channel list, units, sample rate, normalization stats

If `+write_triton=true`, also writes a Triton model store under
`triton_models/<model_name>/`.

**Files**:
- Code: `serving/{onnx_export,model_card_writer,feature_schema}.py`, `serving/triton.py`
- Doc: `docs/lift_to_mlops.md`

---

## Phase 6 — Multi-task heads (fault classifier + RUL + mode predictor)

**Proves**: one Sat-TSFM backbone, four task heads sharing the encoder.
This is "foundation model" in the literal sense.

### Run it

```bash
# CPU smoke
dgx-ts train experiment=phase6_multitask

# Real run with all 4 heads on A5000 / H200
dgx-ts train experiment=phase6_multitask \
    dataset=presets/leo_eps_full_24h \
    model=sat_tsfm_medium \
    trainer=a5000 \
    trainer.max_epochs=50
```

**Output**: MLflow run with per-task validation metrics
(`fault_classifier.acc`, `rul_regressor.mae_log_s`, `mode_predictor.acc`)
in `FitResult.metadata["per_task_metrics"]`.

**Key flag**: `dataset.emit_multitask_labels=true` (required — turns on the
multi-task label generator that emits `aux_labels` on every window).

**Files**:
- Config: `configs/experiment/phase6_multitask.yaml`
- Heads: `models/heads/{fault_classifier,rul_regressor,mode_predictor}.py`
- Wrapper: `models/from_scratch/sat_tsfm_multitask.py`

---

## Phase 7 — Explanation layer (attribution + cascade walker + report)

**Proves**: when AD fires, automatically explain which channels and
timesteps drove the score, walk the coupling graph for upstream causes,
and emit a Markdown report.

### Run it

```bash
# Against a previous MLflow run (top-K most anomalous windows)
dgx-ts explain --run <mlflow_run_id>

# Or specify which experiment to re-fit + explain
dgx-ts explain --run <mlflow_run_id> --top-k 5 --output-dir runs/explanations
```

**Output**: `runs/explanations/window_<start>.md` per window — a
structured Markdown with:
- Top-K driving channels (by Integrated Gradients or attention rollout)
- Upstream cascade path through the coupling graph
- Per-channel score timeline plot
- "Physics covered: yes/no" flag if a PINN was used

**Files**:
- Code: `explanation/{attribution,cascade_walker,report_writer,visualize}.py`
- CLI: `cli/explain.py`

### Sparse Autoencoders (interpretability sprint — Wave 1 shipped)

**Proves**: a wide TopK Sparse Autoencoder trained on frozen Sat-TSFM
encoder activations learns interpretable dictionary features, giving
operators a named vocabulary for "what the model is thinking about."

Currently a library only — no CLI yet. Activation capture is Wave 2.

```python
import numpy as np
from dgx_ts_lab.explanation.sae import TopKSAE, train_sae
from dgx_ts_lab.explanation.sae.train import SAETrainingConfig

# Pretend `acts` is (N, 256) captured from a frozen Sat-TSFM small.
acts = np.random.randn(10_000, 256).astype("float32")

sae = TopKSAE(d_input=256, d_dict=2048, k=32)   # 8x over-complete dict
history = train_sae(
    sae, activations=acts,
    config=SAETrainingConfig(n_epochs=20, batch_size=256),
    device="cpu",
)
print("final recon loss:", history.recon_loss[-1])
print("dead atom fraction:", history.dead_atom_fraction[-1])
```

**Sizing profiles**:

| Config | d_input | d_dict | k | Tier |
|---|---:|---:|---:|---|
| `configs/sae/topk_sae_small.yaml`  | 256 | 2048 (8x)  | 32 | CPU / RTX 3080 |
| `configs/sae/topk_sae_medium.yaml` | 512 | 8192 (16x) | 64 | A5000 / H200 |

**What's coming (Wave 2)**: activation-capture hooks that freeze a
Sat-TSFM checkpoint, stream data through, and dump `(N, d_model)`
activations to parquet. Then feature interpretation (top-K activating
windows per atom).

**Files**:
- Model: `explanation/sae/sae.py` (`TopKSAE`)
- Training: `explanation/sae/train.py` (`train_sae`, `SAETrainingConfig`, `SAETrainingHistory`)
- Configs: `configs/sae/topk_sae_{small,medium}.yaml`
- Design + refs: `packages/dgx_ts_lab/src/dgx_ts_lab/explanation/sae/README.md`

---

## Phase 8 — Cyber AD (command sequences + operator fingerprinting)

**Proves**: discrete command-token sequences and operator activity windows
are new `TelemetryDataset` types that the same trainer handles.

### Run it

```bash
# Full Phase 8 cyber bake-off
dgx-ts benchmark experiment=phase8_cyber trainer=a5000

# Just the sequence transformer (BERT-MLM over commands)
dgx-ts train experiment=phase8_cyber \
    "suite.detectors=[{key: sequence_transformer, params: {}}]"

# Just operator fingerprinting
dgx-ts train experiment=phase8_cyber \
    "suite.detectors=[{key: operator_fingerprint, params: {}}]"
```

**Output**: leaderboard + per-detector npz arrays.

**Files**:
- Datasets: `datasets/cyber/{command_sequence,activity_window,side_channel}.py`,
  `datasets/synthetic/cyber/{command_sequence_gen,operator_traffic_gen}.py`
- Models: `models/from_scratch/sequence_transformer.py`,
  `models/behavior/operator_fingerprint.py`

---

## Phase 9 — Physics-informed AD (thermal + ADCS PINNs)

**Proves**: hand-rolled differentiable physics models combine with
transformer detectors; three SO(3) integrator variants for ADCS.

### Run it

```bash
# Thermal + ADCS PINN bake-off
dgx-ts benchmark experiment=phase9_pinn_bakeoff trainer=a5000

# Just thermal PINN training
dgx-ts train experiment=phase9_pinn_bakeoff \
    model=thermal_pinn_torch
```

**Output**: trained PINN + AD scores. The thermal PINN's physics loss
component appears in MLflow alongside the standard recon/loss curves.

**Files**:
- Configs: `configs/model/{thermal_pinn_torch,adcs_pinn}.yaml`,
  `configs/experiment/phase9_pinn_bakeoff.yaml`
- Code: `models/physics/{thermal_pinn,adcs/}.py`, `models/physics/adcs/{integrators,state}.py`

---

## Phase 10 — Multi-modal foundation (telemetry + commands + logs)

**Proves**: cross-modal MAE pretraining over three aligned streams;
fault-coincident anomalies surface across modalities.

### Run it

```bash
# CPU smoke (~30 s)
dgx-ts train experiment=phase10_multimodal

# Production multi-modal pretrain on DGX
dgx-ts train experiment=dgx_showcase_multimodal     # FSDP 8x H200, large model
```

**Output**: trained `SatMultiModalDetector` with per-modality
reconstruction heads. Score timeline now reflects the *max* per-step
error across all three modality reconstructions.

**Files**:
- Configs: `configs/model/sat_multimodal_{small,medium,large}.yaml`,
  `configs/dataset/multimodal_synth.yaml`,
  `configs/experiment/phase10_multimodal.yaml`
- Code: `datasets/multimodal/`, `models/from_scratch/sat_multimodal.py`

---

## Phase 11 — LLM ops co-pilot (4 backends + RAG + tools)

**Proves**: same co-pilot code runs against Anthropic / vLLM / Ollama /
llama.cpp via the `LLMBackend` Protocol.

### Run it

```bash
# Smoke test (no SDK, no network needed)
dgx-ts copilot --backend mock

# Anthropic — needs ANTHROPIC_API_KEY in env
dgx-ts copilot --backend anthropic

# Local Ollama (workstation tier)
ollama serve &
ollama pull llama3.1:8b
dgx-ts copilot --backend ollama --model llama3.1:8b

# Local vLLM (server / DGX tier)
bash scripts/setup_vllm_server.sh /data/llm_weights/Llama-3.1-70B-Instruct 4 8000
dgx-ts copilot --backend vllm --model meta-llama/Llama-3.1-70B-Instruct \
    --base-url http://localhost:8000/v1

# Air-gap CPU (llama-cpp + GGUF)
dgx-ts copilot --backend llama_cpp --model data/llm_weights/mistral-7b.gguf
```

**With RAG + model card**:

```bash
dgx-ts copilot --backend anthropic \
    --procedures docs/procedures/ \
    --model-card runs/last/model_card.yaml
```

**Files**:
- Code: `llm/{backend,anthropic_backend,vllm_backend,ollama_backend,llama_cpp_backend,_mock_backend,factory,rag,telemetry_tools,copilot,report_generator,procedure_synth}.py`
- Configs: `configs/llm/{anthropic,vllm_llama70b,vllm_mistral_8x22b,ollama_llama8b,llama_cpp_mistral7b_q4}.yaml`
- Doc: `docs/llm_ops_copilot.md`

---

## Visualization layer

**Proves**: every benchmark output is plottable into presentation-grade
ROC / PR / AUC figures without re-running the model.

### Run it

```bash
# Render all standard figures (ROC + PR per dataset+split + AUC bar)
dgx-ts viz --benchmark-dir benchmark_reports/<suite_name>

# SVG output for vector editing
dgx-ts viz --benchmark-dir benchmark_reports/<name> --format png,svg

# Just one split
dgx-ts viz --benchmark-dir benchmark_reports/<name> --splits val
```

**Output**: `benchmark_reports/<name>/figures/`:
- `roc__<dataset>__<split>.{png,svg}`
- `pr__<dataset>__<split>.{png,svg}`
- `auc_bar_val.{png,svg}`

**Files**:
- Code: `evaluation/visualize.py` (5 plot helpers + report bundle)
- CLI: `cli/visualize.py`

---

## Forecasting + RUL bake-off (in progress — W1 shipped)

**Proves**: separate leaderboard for forecasting (multi-step-ahead)
and RUL (remaining useful life) tasks. This is where Prophet, ETS, and
the `RULRegressorHead` finally get numbers.

**Current status**:

| Wave | Deliverable | Status |
|---|---|---|
| W1 | Scoping doc + metrics modules | **Shipped** |
| W2 | Forecasting dataset wrapper + battery/fuel scenarios | Pending |
| W3 | Prophet + ETS adapters | Pending |
| W4 | `dgx-ts forecast-bench` CLI + viz | Pending |
| W5 | Tests + cookbook wrap-up | Pending |

### What you can use TODAY (W1)

Import the metrics modules and score any array of forecasts / RUL
predictions:

```python
import numpy as np
from dgx_ts_lab.evaluation import forecasting_metrics as fm
from dgx_ts_lab.evaluation import rul_metrics as rm

# ── forecasting scorecard ─────────────────────────────
y_true = np.random.randn(20, 8).astype("float32")     # (N windows, H horizon)
y_pred = y_true + 0.1 * np.random.randn(20, 8).astype("float32")
y_train = np.random.randn(200).astype("float32")      # in-sample for MASE scale

# Optional probabilistic:
levels = np.array([0.1, 0.5, 0.9])
y_q = np.stack([y_pred - 0.5, y_pred, y_pred + 0.5], axis=-1)  # (N, H, Q)

bundle = fm.score_bundle(
    y_true, y_pred, y_train=y_train,
    y_quantiles=y_q, quantile_levels=levels,
)
# -> {mae, rmse, smape, mase, crps, pinball_q10/50/90, coverage_80}

# ── RUL scorecard ─────────────────────────────────────
y_true_rul = np.array([30.0, 25.0, 20.0, 15.0, 10.0])  # days remaining
y_pred_rul = np.array([32.0, 28.0, 19.0, 12.0, 11.0])

rul_bundle = rm.score_bundle(y_true_rul, y_pred_rul, tolerances=(1, 7, 30))
# -> {nasa_s_score, mae, rmse, early_mae, late_mae, late_fraction,
#     hit_rate_tol_1, hit_rate_tol_7, hit_rate_tol_30}
```

### Data strategy (locked in W1)

| Task | Real data OK? | Why |
|---|---|---|
| Forecasting | ✅ YES | OPS-SAT (~4 mo) + NASA (weeks/ch) both work |
| RUL | ❌ NO | No run-to-failure trajectories in real spacecraft data |

RUL evaluation goes against synthetic scenarios only:
- **`battery_soc_degradation`** — 6 months synth, EOL = SoC < 30% (W2)
- **`fuel_mass_projection`** — 2 years synth, EOL = fuel < reserve (W2)

Full design + rationale: [`docs/forecasting_rul_bakeoff.md`](forecasting_rul_bakeoff.md).

**Files (Wave 1)**:
- `packages/dgx_ts_lab/src/dgx_ts_lab/evaluation/forecasting_metrics.py`
  (MAE, RMSE, sMAPE, MASE, pinball, CRPS, coverage)
- `packages/dgx_ts_lab/src/dgx_ts_lab/evaluation/rul_metrics.py`
  (NASA S-score, hit-rate-at-tolerance, early/late split, calibration pairs)

---

## DGX procurement showcase (the headline run)

**Proves**: the 8x H200 box can simultaneously train a 1.5B-param multi-task
foundation model, train a multi-modal cross-modal foundation, serve
Mixtral 8x22B for an ops co-pilot, and run a scripted Q&A demo.

### Run it

```bash
# Full showcase on the DGX (6-8 h end-to-end)
bash scripts/dgx_showcase.sh

# Skip individual steps if iterating:
bash scripts/dgx_showcase.sh --skip-tsfm --skip-llm
bash scripts/dgx_showcase.sh --skip-multimodal --skip-export --skip-copilot

# Specify a non-default Mixtral path
bash scripts/dgx_showcase.sh --mixtral-weights /custom/path/Mixtral-8x22B
```

**Output**: `runs/dgx_showcase/`:
- `SHOWCASE_SUMMARY.md` — per-step wall-clock for the deck
- `01_pretrain_sat_tsfm_xl.log` etc. — full training logs
- `exports/sat_tsfm_xl/` and `exports/sat_multimodal_large/` — ONNX +
  model_card + feature_schema bundles
- `copilot_transcript.md` — recorded operator Q&A session

**Companion procurement-deck figures** (generate locally any time):

```bash
python scripts/build_capability_cliff.py
```

Writes `benchmark_reports/capability_cliff/`:
- `capability_ladder.{png,svg}` — log-scale bars per tier
- `capability_matrix.{png,svg}` — green/red checkbox grid
- `dgx_vs_federated.{png,svg}` — 6-metric NVSwitch-vs-PCIe comparison
- `dual_use_capacity.{png,svg}` — GPU-by-GPU LLM + training allocation

**Files**:
- Configs: `configs/experiment/dgx_showcase.yaml`,
  `configs/experiment/dgx_showcase_multimodal.yaml`
- Scripts: `scripts/dgx_showcase.sh`, `scripts/dgx_showcase_copilot_qna.py`,
  `scripts/build_capability_cliff.py`

---

## Cross-cutting CLI cheat-sheet

| Command | Purpose | Phase intro'd |
|---|---|---|
| `dgx-ts train experiment=<name>` | Train a single detector on a dataset | Phase 0 |
| `dgx-ts benchmark experiment=<name>` | Run cartesian product (detector × dataset × seed) | Phase 2 |
| `dgx-ts synth dataset=<name>` | Generate synthetic dataset to disk as parquet | Phase 1 |
| `pwsh scripts/build_dataset.ps1 <name>` | Materialize one preset if not cached | Phase 1 |
| `pwsh scripts/build_corpus.ps1` | Batch-materialize the 6-mission EPS corpus | Phase A |
| `python scripts/download_ops_sat.py` | Fetch real ESA data (connected machine) | Real-data |
| `dgx-ts export model=<name> dataset=<name> +checkpoint=<path>` | Emit ONNX + model_card + feature_schema | Phase 5 |
| `dgx-ts explain --run <mlflow_id>` | Generate per-window explanation reports | Phase 7 |
| `dgx-ts copilot --backend <name>` | Interactive LLM ops co-pilot REPL | Phase 11 |
| `dgx-ts viz --benchmark-dir <path>` | Render ROC/PR/AUC plots from benchmark output | (viz) |
| `dgx-ts forecast-bench experiment=<name>` | Forecasting+RUL bake-off (coming in W4) | (F+RUL) |

---

## Common Hydra overrides (works for any `dgx-ts train/benchmark`)

| Override | Effect |
|---|---|
| `trainer=cpu` / `rtx3080` / `a5000` / `h200` / `h200_fsdp_8x` | Pick the trainer config |
| `trainer.max_epochs=50` | Override epoch count |
| `trainer.batch_size=128` | Per-device batch size |
| `trainer.window_length=512` | Sample-window length |
| `model=<size>` | Swap to a different model-size variant |
| `dataset=<name>` | Swap to a different dataset |
| `+mode=zeroshot` | Run inference only, no training (foundation models) |
| `suite.seeds=[0,1,2,3]` | Number of seeds (benchmark only) |
| `"suite.detectors=[{key: X, params: {...}}]"` | Override the detector list (quote in PowerShell!) |

---

## Where to look when something breaks

| Symptom | Look at |
|---|---|
| `Dataset not registered` | `packages/dgx_ts_lab/src/dgx_ts_lab/datasets/__init__.py` — does the side-effect import include your subpackage? |
| `Detector not registered` | `packages/dgx_ts_lab/src/dgx_ts_lab/models/from_scratch/__init__.py` |
| `CUDA not available` | Run `python -c "import torch; print(torch.cuda.is_available())"`. If False, see `scripts/install_cuda_torch.{ps1,sh}` |
| `uv sync reverted torch to CPU` | Set `UV_NO_SYNC=1` and use `--no-sync` flag on `uv run` |
| `Hydra changes cwd` | `hydra.job.chdir: false` is set globally in `configs/config.yaml` |
| `Tests can't find package` | Check `packages/<pkg>/pyproject.toml` `[tool.hatch.build.targets.wheel] packages = ["src/<pkg>"]` |
| `Plots don't render` | Check `benchmark_reports/<name>/*.npz` exist — without them `dgx-ts viz` has nothing to plot |
| `ANTHROPIC_API_KEY needed` test failure | Live-API test is gated; ignore the skip, or set the env to run it |
| `dataset=leo_eps_v1` Hydra error | Presets live in a subgroup — use `dataset=presets/leo_eps_v1` |
| `dgx-ts train dataset=presets/…` re-generates every run | That's the point of `presets/`. Switch to `dataset=cached/<name>` after `build_dataset` |
| `parquet_telemetry_corpus` complains about channel mismatch | Corpus members must share the exact channel schema — check `channels.yaml` in each `data/synth/<member>/` |
| OPS-SAT download returns 404 | Zenodo rotated the record — see `docs/ops_sat_provisioning.md` URL-rotation table |

---

## Adding a new experiment

1. Drop a YAML in `configs/experiment/<name>.yaml`
2. Add the entry to the table at the top of this doc
3. If it's a new dataset / model, register it in the relevant
   `__init__.py` so the side-effect import fires
4. Run `dgx-ts train experiment=<name>` once to validate
5. Commit per the [commit-style convention](../README.md) — `feat(phaseN): ...`
