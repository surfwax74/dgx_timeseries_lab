# Phases 6–11 — extending the lab beyond AD

Builds on the five shipped phases. Adds 13 demonstration concepts (the user's A1–A4, B5–B8, C9–C10, D11–D13) grouped by architectural pattern into six new phases. Integrates with the existing protocols, registries, Hydra configs, MLflow tracking, deployment tiers, and the Phase 5 MLOps lift contract.

## Locked decisions (from preceding conversation)

| Decision | Locked choice |
|---|---|
| LLM backend strategy | **Quad-backend** — Claude (Anthropic) + vLLM + Ollama + llama.cpp, all behind `LLMBackend` Protocol. Picked per invocation via config. |
| Phase 6 multi-task architecture | **Shared encoder + joint multi-task loss** — one forward pass, four parallel heads, summed weighted loss. |
| Phase 9 PINN dependency | **Optional Modulus + hand-rolled PINN as default** — air-gap clean out of the box; Modulus is opt-in via `[project.optional-dependencies.physics]`. |
| Phase 10 modalities | **Logs + commands + telemetry** (NO imagery). Three modalities all derived from on-orbit operational data: continuous telemetry channels, discrete command opcodes, semi-structured event log entries. |
| Scope | All 6 phases will be built. |
| Integration | Reuse existing protocols, registries, configs, MLflow, deployment tiers. No new top-level packages — everything goes inside `dgx_ts_lab`. |
| Documentation | README at every new directory level (existing standing rule). |
| Tests | Unit + integration tests for every new component. |

## 13 → 6 phases mapping

| Phase | Items | Architectural pattern | Reuses |
|---|---|---|---|
| **6** | A1 fault diagnosis · A2 RUL · A4 mode transition prediction | Multi-task heads on Sat-TSFM backbone | Phase 4 Sat-TSFM, Phase 2 Fabric loop |
| **7** | A3 root-cause · B7-text anomaly report skeleton | Detector-agnostic explanation layer | Any `AnomalyDetector`, Phase 1 coupling graph |
| **8** | D11 command-seq AD · D12 behavior fingerprint · D13 side-channel AD | Sequence/behavior detector family | Phase 1 dataset Protocol, Phase 2 Fabric loop |
| **9** | C9 thermal FEM · C10 ADCS | PINN extensions (hand-rolled default, Modulus optional) | Phase 4 PINN base |
| **10** | B5 multi-modal foundation | Sat-TSFM extension: telemetry + commands + logs | Phase 4 Sat-TSFM, **Phase 8 command sequences** |
| **11** | B6 RAG + telem co-pilot · B7-full prose generation · B8 procedure synthesis | LLM ops platform with quad backend (Anthropic, vLLM, Ollama, llama.cpp) | Phase 5 model card/export, Phase 7 explanation, Phase 8 commands, Phase 10 multi-modal |

## Recommended build order

**6 → 8 → 7 → 9 → 10 → 11**

Rationale:
- Phase 6 has the highest payback per LoC and validates Sat-TSFM as a true foundation.
- Phase 8 is mostly independent (new dataset Protocol implementations + new sequence detector class).
- Phase 7 follows because the explanation layer is more useful with a wider detector family to explain.
- Phase 9 is independent — slot in whenever the Modulus dependency story is ready.
- Phase 10 is the big lift — open up the multi-modal world.
- Phase 11 ties everything together at the end (consumes outputs from all prior phases).

## Cross-cutting infrastructure (built once, used by multiple phases)

These get built in the earliest phase that needs them and reused later.

| Component | First introduced | Reused by |
|---|---|---|
| `models/heads/` directory (task-head modules) | Phase 6 | Phase 10, Phase 11 |
| `HEAD_REGISTRY` (new registry alongside DETECTOR/DATASET/TRAINER) | Phase 6 | Phase 10 |
| `LLMBackend` Protocol + dual implementations (Anthropic / vLLM-local) | Phase 11 *(could be moved earlier if Phase 7 wants prose generation; current plan ships text-only template in Phase 7)* | Phase 7 (optional upgrade) |
| `explanation/` package (attribution, attention rollout, cascade walker, report writer) | Phase 7 | Phase 11 |
| Updated `Capabilities` fields (`supports_multi_task`, `supports_multimodal_input`, etc.) | Per phase as introduced | All later phases |

---

## Phase 6 — Multi-task Sat-TSFM heads

**Goal**: prove Sat-TSFM is a true foundation by attaching three additional task heads — fault classification, RUL regression, and mode-transition prediction — that share the encoder. One backbone, four tasks (AD + 3 new).

### What gets built

| File | Purpose | LoC |
|---|---|---|
| `models/heads/__init__.py` | New `HEAD_REGISTRY`; head Protocol | 60 |
| `models/heads/fault_classifier.py` | Multi-class head over fault types from `fault_log` | 180 |
| `models/heads/rul_regressor.py` | Survival-loss regression head | 200 |
| `models/heads/mode_predictor.py` | Next-mode classification from `ModeMachine` trace | 150 |
| `models/from_scratch/sat_tsfm_multitask.py` | Wraps `SatTSFMDetector` + composable heads, joint multi-task loss | 300 |
| `datasets/synthetic/layered/labels.py` | Extends the layered synth generator to emit RUL trajectories + mode-transition labels (already emits fault_log) | 200 |
| `configs/model/sat_tsfm_multitask.yaml` + 3 head configs | Hydra composition | 80 |
| `configs/experiment/phase6_multitask.yaml` | Bake-off experiment | 60 |
| `tests/test_phase6_heads.py` | Per-head + joint training tests | 400 |
| `models/heads/README.md` | Pattern docs | 100 |

**~1 700 LoC, 10 files.**

### Integration points

- `HEAD_REGISTRY` joins the existing three (DATASET/DETECTOR/TRAINER) — added to `dgx_ts_core.registry`.
- New `Capabilities.supports_multi_task: bool` flag (default False).
- `SatTSFMMultiTaskDetector` registers as a normal detector with extra task outputs surfaced via `FitResult.metadata["per_task_metrics"]`.
- Per-head ONNX export: each head can be exported individually (Phase 5 contract); multi-task model exports as one ONNX with multiple output nodes.

### Acceptance criteria

1. `dgx-ts train experiment=phase6_multitask trainer=rtx3080` runs end-to-end.
2. Joint multi-task loss decreases monotonically over training.
3. Per-task validation metrics surface in MLflow + run summary.
4. Each head can be disabled via config (`heads.{fault,rul,mode}.enabled: false`) — graceful single-task fallback.
5. Existing Phase 4 Sat-TSFM tests still pass (backbone is unchanged).

### Open architectural choice (asked separately below)

- Shared-encoder multi-task vs separate-head per-task fine-tunes from a frozen backbone.

---

## Phase 7 — Explanation layer (A3 + B7-skeleton)

**Goal**: when AD fires, automatically explain *which channels and timesteps drove the score*, walk the coupling graph to identify upstream causes, and render a structured Markdown report. Detector-agnostic — works with any `AnomalyDetector`.

### What gets built

| File | Purpose | LoC |
|---|---|---|
| `explanation/__init__.py` | Public API | 30 |
| `explanation/attribution.py` | Captum Integrated Gradients + attention rollout, dispatch per detector class | 250 |
| `explanation/cascade_walker.py` | Uses the layered-synth coupling graph (or a learned proxy) to trace upstream channel anomalies | 200 |
| `explanation/report_writer.py` | Markdown template: timestamp, ranked channels, fault category guess, suggested next steps, plot links | 220 |
| `explanation/visualize.py` | Matplotlib stubs for per-channel anomaly score timelines | 150 |
| `cli/explain.py` | `dgx-ts explain --run <mlflow-id> [--window-idx N]` | 120 |
| `tests/test_phase7_explanation.py` | Attribution shape tests, cascade-walker correctness on known synth scenarios | 350 |
| `explanation/README.md` | Pattern docs | 80 |

**~1 400 LoC, 8 files.**

### Integration points

- Reads from any MLflow run that produced AD scores.
- For PINN-wrapped detectors (Phase 4), the attribution layer also reports which channels the physics model explained vs which were neural-residual-driven.
- Output goes into MLflow as a new artifact: `explanation_<window_idx>.md`.
- Phase 11 (LLM co-pilot) consumes these Markdown reports as the structural skeleton for prose generation.

### Acceptance criteria

1. `dgx-ts explain --run <mlflow-id>` produces a Markdown report for the top-K anomalous windows in that run.
2. Attribution methods work for all detectors with `supports_export_onnx=True` (Phase 5 pre-req).
3. Cascade walker correctly identifies upstream channels on synthetic fault scenarios where the coupling graph is known.
4. Reports include a "physics covered: yes/no" flag per channel when the detector is PINN-wrapped.

---

## Phase 8 — Behavior AD for cybersecurity (D11, D12, D13)

**Goal**: treat *command sequences*, *operator activity windows*, and *subsystem side-channel patterns* as new `TelemetryDataset` types — no new detector framework needed beyond a sequence-transformer detector class.

### What gets built

| File | Purpose | LoC |
|---|---|---|
| `datasets/cyber/__init__.py` | Public API | 30 |
| `datasets/cyber/command_sequence.py` | Discrete-token dataset (opcodes + parameters as token stream) | 220 |
| `datasets/cyber/activity_window.py` | Rolling statistical windows over operator activity (login times, command rate, command-set diversity, time-of-day patterns) | 220 |
| `datasets/cyber/side_channel.py` | Generic adapter — takes any subsystem's existing telemetry and reframes it as behavior-fingerprint input | 150 |
| `datasets/synthetic/cyber/__init__.py` | Synthetic generator | 30 |
| `datasets/synthetic/cyber/command_sequence_gen.py` | Realistic command-sequence synth + injection patterns (privilege escalation, command flooding, anomalous timing, replay) | 350 |
| `datasets/synthetic/cyber/operator_traffic_gen.py` | Operator-style activity synth with per-operator behavior fingerprints | 250 |
| `models/from_scratch/sequence_transformer.py` | Encoder for discrete-token sequences (BERT-style, separate from continuous Sat-TSFM) | 350 |
| `models/behavior/operator_fingerprint.py` | Per-operator embedding model; AD = distance from operator's baseline distribution | 250 |
| `models/behavior/__init__.py` + README | Pattern docs | 80 |
| `configs/dataset/cyber/{cmdseq, activity, side_channel}.yaml` | Hydra configs | 90 |
| `configs/model/{sequence_transformer, operator_fingerprint}.yaml` | Configs | 60 |
| `configs/experiment/phase8_cyber.yaml` | Bake-off experiment | 60 |
| `tests/test_phase8_cyber.py` | Tests | 450 |

**~2 600 LoC, 18 files.**

### Integration points

- New datasets register with `DATASET_REGISTRY` like any other.
- Sequence transformer registers with `DETECTOR_REGISTRY`.
- Operator fingerprint model has its own `Capabilities` profile: `requires_pretraining=True`, `supports_streaming=True` (computes embedding online), `output_kind=PER_WINDOW`.
- Side-channel adapter is a generic wrapper that takes any telemetry dataset + a "behavior summary" function, exposes a derived `TelemetryDataset`. Lets us apply behavior-style AD to EPS, ADCS, or any subsystem with no new code.

### Acceptance criteria

1. `dgx-ts benchmark experiment=phase8_cyber` runs all three new detector types against their respective synthetic datasets.
2. Operator-fingerprint model successfully separates two synthetic operators by behavior (ROC-AUC > 0.95 on the contrastive validation set).
3. Command-sequence AD detects privilege-escalation injection at > 0.85 ROC-AUC.
4. Side-channel adapter wrapping the existing LEO EPS preset produces meaningful behavior-style scores — validates that the abstraction works.

---

## Phase 9 — PINN extensions: thermal FEM + ADCS (C9, C10)

**Goal**: bring NVIDIA Modulus into the lab as a Phase 4 PINN extension. Thermal FEM PINN replaces our naive `ThermalResidual`; ADCS PINN provides differentiable rigid-body dynamics for control + AD.

### What gets built

| File | Purpose | LoC |
|---|---|---|
| `models/physics/modulus_thermal_fem.py` | Modulus-trained PINN wrapper, implements `PhysicsModel` Protocol — drop-in for `ThermalResidual` | 300 |
| `models/physics/adcs_dynamics.py` | Differentiable rigid-body dynamics (quaternion attitude + reaction-wheel inertia) | 350 |
| `models/physics/modulus_thermal_train.py` | Training script for the thermal PINN (separate from inference path) | 200 |
| `scripts/train_thermal_pinn.sh` | DGX-friendly launcher for the multi-hour training run | 50 |
| `scripts/train_adcs_surrogate.sh` | Same for ADCS surrogate | 50 |
| `datasets/cyber/...` — already covered above | | |
| `configs/model/thermal_pinn_modulus.yaml` | PINN config | 60 |
| `configs/experiment/phase9_pinn_bakeoff.yaml` | Comparison: naive `ThermalResidual` vs Modulus PINN | 80 |
| `tests/test_phase9_pinn.py` | Tests (with mock Modulus when package absent) | 400 |
| `docs/authoring_pinn.md` | Walkthrough doc | 200 |
| `models/physics/README.md` | Update for Modulus pattern | + 50 |

**~1 700 LoC, 9 files.** Plus an external dependency on NVIDIA Modulus (~5GB), made optional via `pyproject.optional-dependencies.physics`.

### Integration points

- Implements existing `PhysicsModel` Protocol — slots into `PINNResidualDetector` unchanged.
- ADCS dynamics is also exposed as a standalone differentiable simulator usable by future RL work (Phase 12+).
- Test fixtures skip-when-absent when Modulus isn't installed, matching the Moirai/uni2ts pattern from Phase 3.

### Open architectural choice (asked separately below)

- Modulus hard-require vs hand-rolled PyTorch PINN extensions.

### Acceptance criteria

1. `dgx-ts train model=thermal_pinn_modulus dataset=presets/leo_eps_full_24h` runs end-to-end when Modulus is installed.
2. PINN bake-off comparison: Modulus thermal PINN beats naive `ThermalResidual` on ROC-AUC for thermal anomalies (target margin > 0.05).
3. ADCS dynamics is differentiable end-to-end (gradient flows from attitude error back to control inputs).
4. Test suite passes both with and without Modulus installed.

---

## Phase 10 — Multi-modal satellite foundation: logs + commands + telemetry (B5, reformulated)

**Goal**: extend Sat-TSFM to ingest THREE on-orbit operational modalities jointly:

1. **Continuous telemetry** — existing Sat-TSFM input (channels × time)
2. **Discrete commands** — opcode + parameter token sequences (reused from Phase 8)
3. **Semi-structured event logs** — timestamp + severity + source + tokenized message text

One model that understands the whole spacecraft operational state — exactly what every operator stares at across three screens today. No imagery, fully air-gap clean, and nobody is shipping this fusion for satellites.

### What gets built

| File | Purpose | LoC |
|---|---|---|
| `models/from_scratch/sat_multimodal.py` | Extended Sat-TSFM: three input streams (telemetry / command tokens / log tokens), cross-modal attention, per-modality reconstruction heads | 550 |
| `models/from_scratch/_multimodal_blocks.py` | Reusable cross-modal attention + modality-type embeddings + temporal alignment | 250 |
| `datasets/multimodal/__init__.py` | Multi-modal dataset Protocol extension (adds `command_windows()`, `log_windows()` to standard `windows()`) | 80 |
| `datasets/multimodal/synth_multimodal_leo.py` | Synthesizes correlated multi-modal LEO data: telemetry from layered synth + matched command stream (Phase 8 reused) + log events that reference both | 450 |
| `datasets/multimodal/log_event_tokenizer.py` | Tokenizes log lines: severity + source + message-text-as-tokens (vocab learned from corpus) | 220 |
| `datasets/synthetic/logs/__init__.py` | Synthetic event log generator (anomaly-correlated alerts + routine ops messages) | 180 |
| `models/from_scratch/README.md` | Update for multi-modal | + 80 |
| `configs/model/sat_multimodal_{small, medium, large}.yaml` | Size variants | 90 |
| `configs/dataset/multimodal_synth.yaml` | Multi-modal config composing telemetry + commands + logs | 80 |
| `configs/experiment/phase10_multimodal.yaml` | Pretraining + downstream eval | 80 |
| `tests/test_phase10_multimodal.py` | Tests | 500 |

**~2 560 LoC, 11 files.** (Roughly same total LoC, but air-gap clean and operationally more relevant.)

### Why logs + commands + telemetry is the right multi-modal choice

- **Operationally faithful**: matches what real sat-ops engineers actually look at — never just imagery.
- **Cross-modal signal**: anomalies frequently appear in *one* modality first (e.g., command failure logged before telemetry shows degraded state). Cross-modal pretraining learns those leading indicators.
- **Reuses Phase 8 directly**: command-sequence dataset + sequence transformer encoder land in Phase 8, get re-used as the command-modality encoder in Phase 10.
- **Air-gap clean**: zero external image archive dependency. All three modalities synthesizable.
- **Novel**: time-series + token-stream multi-modal foundation models for ops telemetry are essentially unexplored.

### Integration points

- New `Capabilities.supports_multimodal_input: bool` flag.
- New optional methods on multimodal-aware datasets: `command_windows(length, stride)` returning token streams, `log_windows(length, stride)` returning event-log token streams. Single-modality datasets simply omit these (backward-compatible).
- Multi-modal pretraining loss: each modality reconstructed from the others (cross-modal MAE — given telemetry + logs, predict commands; etc.).
- **Phase 8 hard dependency**: the command-sequence dataset and sequence transformer from Phase 8 become foundational inputs to Phase 10's command encoder. Build order locks 8 before 10.
- Phase 6 task heads work on multi-modal embeddings unchanged.

### Acceptance criteria

1. `dgx-ts train experiment=phase10_multimodal trainer=h200` pretrains the multi-modal model end-to-end on synth data.
2. Cross-modal reconstruction works: ablating one modality at inference, the model can predict it from the other two (within tolerance — exact metric per modality type).
3. Multi-modal model beats telemetry-only Sat-TSFM on Phase 6's fault classification task by ≥ 0.05 macro-F1 (showing logs + commands add signal).
4. Cross-modal AD: anomalies injected ONLY in one modality (e.g., logs) get detected via cross-modal reconstruction error in the other modalities. Tests cover each of the three single-modality fault types.
5. Size variants (small / medium / large) all fit their target tier (Phase 4 tier mapping).

---

## Phase 11 — LLM ops co-pilot (B6 + B7-full + B8) with dual backend

**Goal**: interactive AI ops assistant grounded in live telemetry, mission docs, and exported model cards. Supports both **online** (Claude via Anthropic API) and **air-gapped** (Llama / Mistral via vLLM) backends, selectable per invocation.

### What gets built

| File | Purpose | LoC |
|---|---|---|
| `llm/__init__.py` | Public API | 40 |
| `llm/backend.py` | `LLMBackend` Protocol — async `generate()`, `stream()`, `tools_call()` | 150 |
| `llm/anthropic_backend.py` | Claude implementation (anthropic SDK, prompt caching, tool use) | 280 |
| `llm/vllm_backend.py` | vLLM HTTP client backend (OpenAI-compatible API) | 250 |
| `llm/ollama_backend.py` | Ollama HTTP client backend (small model registry, easy DX) | 200 |
| `llm/llama_cpp_backend.py` | llama-cpp-python single-process backend (CPU/GPU, GGUF quantized) | 220 |
| `llm/rag.py` | RAG: FAISS index over mission procedures + retrieval | 250 |
| `llm/telemetry_tools.py` | Tool definitions: `query_telemetry`, `query_anomaly_history`, `lookup_procedure`, `read_model_card` | 300 |
| `llm/report_generator.py` | B7 full: Phase 7 Markdown skeleton + telemetry + recent anomalies → polished prose report | 220 |
| `llm/procedure_synth.py` | B8: natural language → command sequence with simulator validation loop | 280 |
| `llm/copilot.py` | Interactive multi-turn chat orchestrator | 250 |
| `cli/copilot.py` | `dgx-ts copilot [--backend {anthropic,vllm,ollama,llama_cpp}]` REPL | 180 |
| `scripts/download_local_llm_weights.sh` | Connected-machine helper: downloads Llama/Mistral GGUF + safetensors for sneakernet | 100 |
| `scripts/setup_vllm_server.sh` | DGX vLLM launcher | 70 |
| `scripts/setup_ollama_server.sh` | Workstation/server Ollama launcher | 50 |
| `configs/llm/{anthropic, vllm_llama70b, vllm_mistral_8x22b, ollama_llama8b, llama_cpp_mistral7b_q4}.yaml` | Backend configs covering each tier | 150 |
| `tests/test_phase11_llm.py` | Backend abstraction tests with mock LLM + per-backend integration tests | 500 |
| `llm/README.md` | Pattern + quad-backend docs | 250 |
| `docs/llm_ops_copilot.md` | User-facing guide + per-backend selection matrix | 300 |

**~3 840 LoC, 19 files.** Plus weights provisioning (3–200 GB depending on which backend × which model).

### Integration points

- `LLMBackend` Protocol is the central abstraction. Four implementations live under `llm/`:
  - **AnthropicBackend** — requires `ANTHROPIC_API_KEY` env, online only. Uses prompt caching + tool use natively. Best for dev workstations + non-air-gap deployments where Claude quality matters most.
  - **VLLMBackend** — connects to a running vLLM server (`scripts/setup_vllm_server.sh` launches it). Best for H200 / A5000×N tiers; production-grade throughput.
  - **OllamaBackend** — connects to a local Ollama daemon. Best for RTX 3080 / A5000 single-GPU dev; easiest setup.
  - **LlamaCppBackend** — single-process, in-tree. Best for CPU-only or RTX 3080 with small quantized models (Mistral 7B Q4); no separate server needed.
- Switch backends per invocation: `dgx-ts copilot --backend anthropic` / `--backend vllm` / `--backend ollama` / `--backend llama_cpp`.
- Per-tier defaults in the deployment playbook: laptop → llama_cpp; RTX 3080 → ollama; A5000 → vllm; H200 → vllm; non-air-gap dev → anthropic.
- All other phases' artifacts feed in:
  - Phase 5 model_card.yaml → consumed by `read_model_card` tool
  - Phase 7 explanation reports → consumed by `lookup_procedure` and `report_generator`
  - Phase 10 multi-modal foundation → potentially used as a structured-data encoder before the LLM
- Air-gap: vLLM server lives on the DGX; no external connections. Documented as part of the DGX deployment playbook.

### Open architectural choice (asked separately below)

- Local LLM serving stack: vLLM vs Ollama vs llama.cpp.

### Acceptance criteria

1. `dgx-ts copilot --backend anthropic` runs against Claude API end-to-end (when key is present).
2. `dgx-ts copilot --backend vllm` runs against a local vLLM server with a Llama or Mistral model end-to-end.
3. Same prompts produce sensible (if not identical) answers across both backends.
4. Tool-use harness exposes `query_telemetry`, `query_anomaly_history`, `lookup_procedure`, `read_model_card` — each tested with mock LLM.
5. Report generator (B7) consumes a Phase 7 explanation skeleton + recent telemetry windows and emits a polished operator-facing Markdown report.
6. Procedure synthesizer (B8) successfully maps natural-language requests to valid command sequences for at least 5 representative scenarios; invalid syntheses are caught by simulator validation and corrected via a tool-call loop.

---

## Integration with existing platform — concrete touchpoints

### New things added to `dgx_ts_core`

- `Capabilities` gains: `supports_multi_task`, `supports_multimodal_input` (existing fields unchanged).
- `registry.py` gains `HEAD_REGISTRY` (Phase 6).
- No new Protocols — heads, sequence detectors, LLM backends all live in `dgx_ts_lab`.

### New things added to `dgx_ts_lab`

- `models/heads/` (Phase 6) — new subpackage.
- `models/behavior/` (Phase 8) — new subpackage.
- `datasets/cyber/` (Phase 8) — new subpackage.
- `datasets/synthetic/cyber/` (Phase 8) — new synth subpackage.
- `datasets/multimodal/` (Phase 10) — new subpackage.
- `explanation/` (Phase 7) — new top-level subpackage.
- `llm/` (Phase 11) — new top-level subpackage.
- `serving/` updated for multi-output + multi-modal ONNX export (Phase 10 follow-ups).

### CLI subcommands added

- `dgx-ts explain` (Phase 7)
- `dgx-ts copilot` (Phase 11)
- Existing `train`, `synth`, `benchmark`, `export` continue to work for all new detector types.

### Configs added

- `configs/model/` — heads, multitask, sequence, multi-modal variants
- `configs/dataset/` — cyber variants, multimodal
- `configs/llm/` (Phase 11) — backend configs
- `configs/experiment/` — one bake-off per new phase

### Tier compatibility (additions to the hardware matrix)

| Model | CPU | RTX 3080 | A5000 | A5000×8 | H200 | 8×H200 |
|---|---|---|---|---|---|---|
| Sat-TSFM multitask (Phase 6, ~120M) | ✗ | ⚠ tight | ✓ | ✓ | ✓ | ✓ |
| Sequence transformer cyber (Phase 8) | ⚠ slow | ✓ | ✓ | ✓ | ✓ | ✓ |
| Operator fingerprint (Phase 8) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Modulus thermal PINN (Phase 9) | ✗ | ⚠ small | ✓ | ✓ FSDP for training | ✓ | ✓ |
| Multi-modal Sat-TSFM small (Phase 10, ~80M) | ✗ | ⚠ tight | ✓ | ✓ | ✓ | ✓ |
| Multi-modal Sat-TSFM medium (~400M) | ✗ | ✗ | ⚠ tight | ✓ FSDP | ✓ | ✓ |
| Multi-modal Sat-TSFM large (~1.5B) | ✗ | ✗ | ✗ | ⚠ FSDP | ⚠ tight | ✓ FSDP |
| Llama 8B (Phase 11 local) | ⚠ very slow | ⚠ Q4 quantized only | ✓ Q4/Q8 | ✓ FP16 | ✓ FP16 | ✓ FP16 |
| Llama 70B (Phase 11 local) | ✗ | ✗ | ⚠ Q4 only | ✓ FSDP Q4 | ⚠ Q4 only | ✓ FSDP FP16 |
| Mistral 8×22B (Phase 11 local) | ✗ | ✗ | ✗ | ⚠ Q4 FSDP | ⚠ Q4 only | ✓ FSDP FP16 |

(Will update `docs/deployment/hardware_compatibility_matrix.md` as each phase lands.)

## Total scope

| Phase | Files | LoC | Complexity |
|---|---:|---:|---|
| 6 | 10 | 1 700 | medium |
| 7 | 8 | 1 400 | medium |
| 8 | 18 | 2 600 | high |
| 9 | 9 | 1 700 | medium (Modulus optional) |
| 10 | 11 | 2 560 | high |
| 11 | 19 | 3 840 | high |
| **Total** | **75** | **~13 800** | — |

For comparison: phases 0–5 together were ~190 files and ~17 750 LoC. This roughly doubles the codebase.

## Build-order reminders

1. Phase 6 first — fastest payback, validates Sat-TSFM as a foundation.
2. Phase 8 second — independent, broadens framework reach into cyber.
3. Phase 7 third — explanation is more useful with more to explain.
4. Phase 9 fourth — independent, slot in once Modulus question is settled.
5. Phase 10 fifth — multi-modal opens new territory.
6. Phase 11 last — ties everything together via LLM-driven UX.

## Open architectural questions — RESOLVED

| Question | Resolution |
|---|---|
| Phase 6 multi-task strategy | Shared encoder + joint multi-task loss |
| Phase 9 Modulus dependency | Optional Modulus + hand-rolled PINN as default |
| Phase 10 modality choice | Logs + commands + telemetry (no imagery) |
| Phase 11 local LLM serving stack | All four backends behind `LLMBackend` Protocol (Anthropic + vLLM + Ollama + llama.cpp) |

## Deviations log (post-execution)

*(empty — populate as each phase ships with anything that diverged from the locked plan)*
