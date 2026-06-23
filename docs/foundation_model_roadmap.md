# Foundation Model Roadmap

Planning doc for time-series + LLM foundation models and interpretability
methods we want to add. Each entry has the HF repo, size, what it
unlocks, integration effort, and air-gap weight notes so someone picking
this up next quarter can act without rediscovering the context.

**Status**: scoped, not started. Nothing in this doc is in `models/README.md`
yet — it's the next-phase backlog.

---

## Quick reference

| Priority | Addition | Family | HF repo | Size | Effort | Unlocks |
|---:|---|---|---|---:|:---:|---|
| **1** | [TimesFM](#timesfm-google) | TS foundation | `google/timesfm-2.0-500m-pytorch` | 500M | 1 day | Closes the most-cited TS-FM gap; expected on any leaderboard |
| **2** | [TTM (Tiny Time Mixer)](#ttm-tiny-time-mixer-ibm) | TS foundation | `ibm-granite/granite-timeseries-ttm-r2` | 1–5M | 1 day | Fast LoRA on RTX 3080; fills the "tiny foundation" tier |
| **3** | [TimeMoE](#timemoe) | TS foundation | `Maple728/TimeMoE-200M` | 200M (MoE) | 2 days | Pairs with our `SubsystemMoE`; demonstrates we know the MoE landscape |
| **4** | [Granite Code](#granite-code-ibm-llm-co-pilot) | LLM (Phase 11) | `ibm-granite/granite-3.2-8b-instruct` | 8B / 20B / 34B | **1 hour** (config only) | Better JSON-mode reliability for procedure synthesizer; Apache-2.0 licensing for air-gap |
| **5** | [Phi-4](#phi-4-microsoft-llm-co-pilot) | LLM (Phase 11) | `microsoft/phi-4` | 14B | **1 hour** (config only) | "Small but smart" co-pilot — fits A5000 24 GB in bf16 |
| **6** | [Sparse Autoencoders](#sparse-autoencoders-sae-phase-7-interpretability) | Phase 7 interpretability | (multiple) | varies | 3–5 days | Modern interpretability layer beyond attention rollout; explains what the encoder *learned*, not just what it attended to |
| **7** | [ETS / Holt-Winters](#ets--holt-winters-classical-baseline) | Classical baseline | `statsmodels` | n/a | **0.5 day** | Stronger classical baseline than `rolling_mean` for trend-and-seasonality channels; no heavyweight deps; explicitly preferred over Prophet (see [Prophet deferral note](#why-prophet-was-deferred)) |

---

## TimesFM (Google)

**Repo:** `google/timesfm-2.0-500m-pytorch`
**Architecture:** Decoder-only patched transformer.
**License:** Apache 2.0.
**Why now:** TimesFM is the single most-cited TS foundation model of
2024–25 and is conspicuously absent from our bake-off. Reviewers
specifically look for it.

**Implementation pattern** — mirrors `models/foundation/chronos.py` almost 1:1:

```
packages/dgx_ts_lab/src/dgx_ts_lab/models/foundation/timesfm.py
    class TimesFMDetector:
        capabilities = Capabilities(
            requires_pretraining=False,    # zero-shot path
            supports_streaming=False,
            supports_multivariate=True,    # per-channel forecasts
            native_context_len=512,
            output_kind=OutputKind.PER_STEP,
            supports_peft=True,             # LoRA path
            supports_export_onnx=False,
        )

        def fit(self, dataset, mode, config):
            # mode == ZEROSHOT: just calibrate threshold
            # mode == FINETUNE: LoRA via peft, reuse _peft_helper from Chronos
            ...

        def compute_score_batch(self, batch):
            # forecast_horizon_1 with the loaded model, residual = score
            ...

configs/model/timesfm_zero.yaml
configs/model/timesfm_lora.yaml
```

**Air-gap weights**: `huggingface-cli download google/timesfm-2.0-500m-pytorch`
on a connected box, sneakernet the resulting `.safetensors`. ~1 GB.
Register via the existing `scripts/register_foundation_models.py` pattern.

**Acceptance**: appears in `dgx-ts benchmark experiment=phase3_bakeoff`
output, beats `rolling_mean` on at least 3 of the 4 SMAP test channels.

---

## TTM (Tiny Time Mixer, IBM)

**Repo:** `ibm-granite/granite-timeseries-ttm-r2` (also `r1` for comparison)
**Architecture:** MLP-Mixer-based, NOT a transformer. Interesting
architectural diversity vs. everything else in the bake-off.
**License:** Apache 2.0.
**Why now:** Fills the "tiny foundation model" tier we don't have.
Currently every foundation model in our lab needs an A5000+ to fine-tune;
TTM does LoRA fine-tuning on a 3080 in minutes. Strong story for the
"workstation-only" deployment narrative.

**Implementation pattern**: same as Chronos. The MLP-Mixer block is
trivially different from a transformer — `forward()` is a few stacked
`nn.Linear` + token-mixing + channel-mixing operations.

```
packages/dgx_ts_lab/src/dgx_ts_lab/models/foundation/ttm.py
    class TTMDetector:
        # supports_peft=True, native_context_len=512, 1-5M params
        # Output: per-step reconstruction error
```

**Air-gap weights**: ~30 MB. Trivially sneakernet-able.

**Demo angle for procurement**: "On the 3080 tier, you fine-tune TTM in
3 minutes and get production-quality detection. The DGX is for the big
foundation models — TTM proves the lab works at every tier."

---

## TimeMoE

**Repo:** `Maple728/TimeMoE-200M` (also `Maple728/TimeMoE-50M` for comparison)
**Architecture:** Mixture-of-Experts transformer. 200M total params, ~50M
active per forward pass.
**License:** Apache 2.0.
**Paper:** Shi et al., *"Time-MoE: Billion-Scale Time Series Foundation
Models with Mixture of Experts"* (NeurIPS 2024).
**Why now:** Pairs *directly* with our hand-rolled `SubsystemMoE`
detector. The story: "we have both a generic MoE (TimeMoE) and a
subsystem-routed MoE (our `SubsystemMoE`); here's how they compare on the
LEO EPS dataset." That's the kind of head-to-head a reviewer remembers.

**Implementation pattern**: same as Chronos, but the model has internal
expert routing that needs to be exposed in the adapter. Two considerations:

1. **vLLM-style continuous batching is not free here** — MoE inference
   has uneven expert utilization. Document the throughput characteristic
   in `models/foundation/README.md`.
2. **PEFT/LoRA on MoE** — only adapter weights on attention matter; the
   experts themselves stay frozen. Document this in the LoRA config.

**Air-gap weights**: ~800 MB for the 200M model; 200 MB for the 50M.

**Demo angle**: side-by-side AUC and inference-throughput comparison
between `TimeMoEDetector(200M)` and `SubsystemMoEDetector(200M)` on the
83-channel LEO EPS preset. Adds a "MoE vs MoE" panel to the procurement
deck.

---

## Granite Code (IBM, LLM co-pilot)

**Repo:** `ibm-granite/granite-3.2-8b-instruct` (code-tuned variants:
`granite-3.2-3b-instruct`, `granite-20b-code-instruct`, `granite-34b-code-instruct`)
**Architecture:** Standard decoder-only transformer.
**License:** Apache 2.0 — critical for air-gap (some Llama variants have
acceptable-use restrictions that legal flags in classified deployments).
**Why now:** The Phase 11 `ProcedureSynthesizer` relies on strict
JSON-mode output. Granite-Code is specifically tuned for code/structured
output and is **substantially more reliable at JSON-only responses** than
the same-size Llama or Mistral. Failure rate of the procedure synth's
retry loop should drop meaningfully.

**Implementation effort**: **zero new code**. Phase 11's `LLMBackend`
Protocol means a new model is a YAML config:

```yaml
# configs/llm/vllm_granite_8b_code.yaml
kind: vllm
model_id: ibm-granite/granite-3.2-8b-instruct
base_url: http://localhost:8000/v1
api_key: EMPTY
timeout_s: 120.0

# configs/llm/vllm_granite_34b_code.yaml
kind: vllm
model_id: ibm-granite/granite-34b-code-instruct
base_url: http://localhost:8000/v1
api_key: EMPTY
timeout_s: 180.0
```

**Air-gap weights**: 8B is ~16 GB bf16, ~8 GB INT8 — fits a single H200
comfortably. 34B is ~68 GB bf16, ~34 GB INT8 — needs TP=2 on H200 or fits
in 1 H200 at INT4.

**Deployment recommendation**:
- **Single H200 + Granite 8B-Code INT8**: ~10 GB used, leaves 130 GB free
  for training jobs on the same GPU.
- **A5000 (24 GB) + Granite 8B-Code INT8**: ~10 GB used, ~14 GB free for
  KV cache (big batches). Sweet spot for procedure synthesis on
  workstation tier.

**Demo angle**: A/B test the procedure synthesizer with Llama 8B vs.
Granite 8B-Code on a battery of 20 NL → command-sequence requests.
Granite's parse-success rate (no retry needed) should be ~15-25 percentage
points higher. That's a slide.

**Demo command**:

```bash
# Spin up vLLM with Granite-Code 8B
bash scripts/setup_vllm_server.sh /data/llm_weights/granite-3.2-8b-instruct 1 8000

# Run a scripted procedure-synth benchmark against both backends
python scripts/dgx_showcase_copilot_qna.py \
    --backend vllm \
    --model ibm-granite/granite-3.2-8b-instruct \
    --base-url http://localhost:8000/v1 \
    --output runs/copilot_granite_8b.md
```

---

## Phi-4 (Microsoft, LLM co-pilot)

**Repo:** `microsoft/phi-4`
**Architecture:** Standard decoder-only transformer with synthetic-data
training emphasis.
**License:** MIT.
**Why now:** Best small-model reasoning of late 2024 / early 2025. 14B
params hits the "small but smart" niche — between Llama-8B (too weak for
multi-step ops reasoning) and Llama-70B (overkill for most workstation
deployments). Fits A5000 24 GB at bf16 with ~10 GB free for KV cache.

**Implementation effort**: **zero new code**. New YAML configs:

```yaml
# configs/llm/vllm_phi4.yaml
kind: vllm
model_id: microsoft/phi-4
base_url: http://localhost:8000/v1
api_key: EMPTY
timeout_s: 120.0

# configs/llm/ollama_phi4.yaml
kind: ollama
model_id: phi4:14b
base_url: http://localhost:11434
timeout_s: 120.0
```

**Air-gap weights**: ~28 GB bf16, ~14 GB INT8, ~7 GB INT4.

**Deployment recommendation**:
- **A5000 (24 GB) tier**: Phi-4 INT8 at ~14 GB + KV cache. Best
  agentic-LLM choice for this tier — Llama-8B is too weak, Llama-70B
  won't fit.
- **RTX 3080 (10 GB) tier**: Phi-4 INT4 just fits (~7 GB weights + small
  KV cache for context ~4 K).

**Demo angle**: extends the dual-use capacity figure (`build_capability_cliff.py`)
with a new row showing Phi-4 on the A5000 tier as a "single workstation
agentic ops" option that doesn't need the DGX.

---

## Sparse Autoencoders (SAE, Phase 7 interpretability)

**Reference implementations:**
- `EleutherAI/sae` — general SAE training library
- `goodfire-ai/sae` — production-quality SAE tooling
- Anthropic's open SAE work (Sonnet 3.5 interpretability paper, 2024)

**License:** Apache 2.0 / MIT (varies by repo).

**What it is:** A sparse autoencoder trained on the hidden activations of
a transformer encoder. The SAE decomposes each activation into a sparse
combination of human-interpretable "features." For our Sat-TSFM
encoder, this would mean discovering features like "post-eclipse warm-up
transient" or "battery sag during high-load mode" — interpretable axes
of variation in the latent space.

**Why now:** Phase 7's current attribution layer uses Captum Integrated
Gradients + attention rollout. Those tell you *which input timesteps and
channels* drove a score. SAEs tell you *what abstract feature* the model
detected. For the procurement story this is the difference between
"channel `bus_v` had high gradient" (current Phase 7) and "the model
detected the 'bus undervoltage during eclipse exit' feature" (SAE-based).

**Implementation pattern** — new subpackage under `explanation/`:

```
packages/dgx_ts_lab/src/dgx_ts_lab/explanation/sae/
    __init__.py
    sae_trainer.py        # Train an SAE against a captured-activations dataset
    sae_attributor.py     # Use trained SAE features as attribution targets
    feature_atlas.py      # Discover + name features by ranking activations
                          # against per-window labels (fault types, modes)
configs/explanation/sae/
    sat_tsfm_sae.yaml     # SAE training config: encoder layer to probe,
                          # dictionary size, sparsity penalty
```

**Effort breakdown** (3–5 days):

| Step | Effort |
|---|---|
| Capture activations from a trained Sat-TSFM encoder over a dataset | 0.5 day |
| Wire one of the open SAE libraries (`EleutherAI/sae` is cleanest) | 1 day |
| Train an SAE on those activations (CPU OK for small dict; H200 for big) | 1 day |
| Build the feature-atlas: rank SAE features by correlation with known fault types | 1 day |
| Integrate SAE attribution into the Phase 7 report writer alongside IG | 1 day |
| Tests + README | 0.5 day |

**Blockers**: SAE training is genuinely a research activity — there's no
"plug and play." Features won't be cleanly interpretable on the first
try. Plan for an iteration cycle on dictionary size, sparsity penalty,
and which encoder layer to probe.

**Demo angle**: a slide showing the top 10 SAE features the model
learned, each captioned with its discovered concept (e.g., "Feature 47:
fires during EPS bus voltage recovery after eclipse exit"). This is the
*"mechanistic interpretability"* story that's currently hot in ML
research and which procurement audiences will recognize as
state-of-the-art. Differentiates the lab from anyone else doing AD.

---

## ETS / Holt-Winters (classical baseline)

**Library:** `statsmodels.tsa.holtwinters.ExponentialSmoothing` (per-channel
parametric model) + `statsmodels.tsa.statespace.exponential_smoothing`
(state-space form, supports streaming via Kalman filter).
**Backing organization:** `statsmodels` (Apache 2.0, ~14k GitHub stars,
maintained since 2009).
**Why now:** `rolling_mean` is the current classical baseline. It nails
spike anomalies but is silent on slow drift and seasonality. ETS adds a
credible second classical baseline that handles trend + seasonality
without pulling in transformer machinery — sets a *fair* bar before the
neural detectors take the stage on slow-changing channels.

**Concretely:** Holt-Winters' triple exponential smoothing
(`SimpleExpSmoothing` / `Holt` / `ExponentialSmoothing` with `trend=` and
`seasonal=` parameters) is the algorithmic ancestor of every modern TS
forecasting method. AD score = standardized residual from one-step-ahead
prediction. The state-space form means online updates are O(1) per step.

**Implementation pattern**:

```python
# packages/dgx_ts_lab/src/dgx_ts_lab/models/baseline/ets.py
@DETECTOR_REGISTRY.register("ets")
def _create(...) -> ETSDetector:
    return ETSDetector(...)

class ETSDetector:
    capabilities = Capabilities(
        requires_pretraining=False,       # online fit, no GPU
        supports_streaming=True,          # state-space form supports it
        supports_multivariate=False,      # per-channel (caller iterates)
        native_context_len=None,          # unbounded
        output_kind=OutputKind.PER_STEP,
        supports_peft=False,
        supports_export_onnx=False,       # statsmodels objects don't trace
    )

    def fit(self, dataset, mode, config):
        # Fit one ExponentialSmoothing per channel.
        # Store fitted states + residual std so .score() is O(1).
        ...

    def score(self, window):
        # One-step-ahead prediction per channel; score = |observed - pred| / sigma.
        # Aggregate per-channel scores to per-step via max.
        ...
```

`configs/model/ets.yaml`:

```yaml
_target_key: ets
trend: add                # 'add' | 'mul' | null
seasonal: add             # 'add' | 'mul' | null
seasonal_periods: 5400    # 90-min LEO orbit at 1 Hz
damped_trend: true
```

**Dependency burden:** statsmodels is already a transitive dep of
scikit-learn (which is already in our deps). No new heavyweight imports,
no native compilation, no air-gap headaches.

**Demo angle:** the bake-off becomes a three-tier ladder of classical
baselines: `rolling_mean` (point spikes) → `ets` (slow trend +
seasonality) → neural detectors (everything else). Audience sees a fair
gradient, and the procurement slide reads "we beat the classics" instead
of "we beat one classic."

**Tests:** parameterize the existing `test_rolling_mean_detector.py` to
also cover `ets` against the layered synth (orbital sinusoid + drift
fault should score detectably; pure Gaussian noise should score near
zero).

### Why Prophet was deferred

Prophet (Meta, originally Facebook 2017) is **not on this roadmap** for
four reasons:

1. **Pulls Stan/cmdstanpy** — a heavy C++ build toolchain that's hostile
   to the air-gap install story.
2. **Univariate-only** and refits per-series. ~10 s per channel; ~14 min
   for the 83-channel LEO EPS preset just for the baseline.
3. **Training-time outlier removal** silently discards the very events
   we want to detect.
4. **Architecturally redundant with ETS** — both decompose into trend +
   seasonality. ETS does it with 50 lines of state-space math; Prophet
   does it with a Stan model. Same algorithmic family, vastly different
   dependency cost.

If a stakeholder specifically asks for Prophet (analyst familiarity, a
specific demo where Prophet has already been adopted), see the
"[Where Prophet might fit in production telemetry](#where-prophet-might-actually-fit-on-real-subsystems)"
section below — it documents the subsystem-by-subsystem assessment so
you can make an evidence-based call.

### Where Prophet might actually fit on real subsystems

The honest version is: Prophet works on **slow, smooth, seasonal**
signals and fails on **fast, noisy, transient** ones. Per spacecraft
subsystem:

| Subsystem / channel class | Prophet would… | Why |
|---|---|---|
| **EPS — battery SoC long-term trend** (averaged hourly+) | **Work as a forecasting model** for capacity degradation prognosis | Orbital cycle + slow capacity loss is the canonical Prophet shape |
| **EPS — solar panel power** (daily averaged) | **Work as a forecasting model** | Strong yearly + seasonal sun-angle pattern; smooth additive |
| **EPS — bus voltage / current at 1 Hz** | **Fail for AD** | Sub-second transients smoothed away; cross-channel coupling invisible |
| **TCS — panel/structural temperatures** (10-min averaged) | **Work as a baseline** for thermal drift | Smooth diurnal-like cycles + seasonal drift |
| **TCS — heater duty cycles** | **Fail** | Discrete on/off state, not additive smooth |
| **ADCS — attitude / gyro / wheel speeds** | **Fail badly** | Sub-second dynamics, tightly coupled control loops; Prophet has no concept of control feedback |
| **PROP — fuel mass remaining** | **Work well** for prognosis | Monotonic slow trend + discrete burn "holidays" — Prophet's holiday mechanism literally fits this pattern |
| **COMMS — daily downlink volume / contact count** | **Work as a baseline** | Operational seasonality (weekly cadence, ground-station availability) |
| **COMMS — link SNR at high rate** | **Fail** | Geometry + atmosphere creates fast variability Prophet can't track |
| **OBDH — command counts, telemetry rates** (daily) | **Work as a baseline** | Operational weekly cadence (weekday vs weekend) |
| **PAYLOAD — instrument duty / data product counts** (daily) | **Work as a baseline** | Mission-cadence seasonality |
| **Mode-machine transitions** | **Fail** | Discrete state, not continuous |
| **Cross-modal (telem + cmd + log)** | **Fail** | Univariate-only by design |

**Net summary**: Prophet would be a reasonable choice for **long-horizon
trend forecasting on aggregated slow channels** — battery degradation
prognosis, fuel-mass projection, daily mission cadence anomalies. It is
a **poor choice for real-time anomaly detection** on 1 Hz raw telemetry,
which is what the rest of this lab is built for.

If a procurement audience asks why Prophet isn't in the bake-off, the
defensible answer is: "Prophet is in the *prognostics* family (predict
months ahead), not the *AD* family (detect now). We benchmark against
the right tools for the AD task. We'd add Prophet to a *separate*
long-horizon-prognostics demo if that becomes part of the scope."

If you want that prognostics demo for the battery-degradation slide, the
right framing is:

| Detector | Task | Horizon |
|---|---|---|
| `patchtst_mae` / `sat_tsfm` / etc. | Real-time AD on 1 Hz raw | seconds to minutes ahead |
| **Prophet** | Long-horizon prognosis on daily aggregates | weeks to months ahead |
| `thermal_pinn_torch` / `battery_residual` | Physics-grounded prediction | minutes to hours ahead |

That's the honest position: don't fight Prophet, scope it to where it's
strong. Adding it as `models/baseline/prophet.py` for the prognostics
use case is ~1 day if and when the scope expands; the deferral above
applies specifically to using it for AD on raw telemetry.

---

## Standard "add a foundation model" pattern (for items 1–3 above)

Each TS foundation model follows the same recipe — established in Phase 3
when we built Chronos / MOMENT / Moirai adapters:

1. **Create `models/foundation/<model>.py`** with a `<Model>Detector` class
   that implements the neural-detector contract (or the lighter detector
   contract for zero-shot-only models).
2. **Reuse `models/foundation/_loader.py`** for sneakernet-aware weight
   loading — it handles HF Hub fallback to local `data/models/<name>/`
   when offline.
3. **Reuse `models/foundation/_peft_helper.py`** for the LoRA path.
4. **Add to `models/foundation/__init__.py`** so the registry decorator
   fires at import time.
5. **Write `configs/model/<name>_zero.yaml`** and `configs/model/<name>_lora.yaml`.
6. **Tests in `packages/dgx_ts_lab/tests/test_foundation.py`** — extend
   the parameterized test that covers Chronos/MOMENT/Moirai today.
7. **Add a row to `packages/dgx_ts_lab/src/dgx_ts_lab/models/foundation/README.md`**
   and to the inventory in `models/README.md`.

Total per-model code: ~250–350 LoC + ~50 LoC tests + 2 YAMLs.

---

## Air-gap weight provisioning workflow

For the three TS foundation models + the two LLMs, the procurement-ready
workflow:

```bash
# On a connected workstation
mkdir -p /tmp/foundation_bundle

# TS foundations
huggingface-cli download google/timesfm-2.0-500m-pytorch \
    --local-dir /tmp/foundation_bundle/timesfm
huggingface-cli download ibm-granite/granite-timeseries-ttm-r2 \
    --local-dir /tmp/foundation_bundle/ttm
huggingface-cli download Maple728/TimeMoE-200M \
    --local-dir /tmp/foundation_bundle/timemoe

# LLMs (these are bigger — split into separate transfers if needed)
huggingface-cli download ibm-granite/granite-3.2-8b-instruct \
    --local-dir /tmp/foundation_bundle/granite-8b
huggingface-cli download microsoft/phi-4 \
    --local-dir /tmp/foundation_bundle/phi-4

# Bundle for sneakernet
cd /tmp
tar czf foundation_bundle.tar.gz foundation_bundle/
sha256sum foundation_bundle.tar.gz > foundation_bundle.tar.gz.sha256

# Sizes (approx):
# timesfm        ~1 GB
# ttm            ~30 MB
# timemoe        ~800 MB
# granite-8b     ~16 GB
# phi-4          ~28 GB
# TOTAL bundle:  ~46 GB compressed
```

On the DGX: untar to `data/models/` and `data/llm_weights/` per the
existing convention. Wire up paths in `configs/model/*.yaml` and
`configs/llm/*.yaml` respectively.

---

## Sequencing recommendation

If you have one engineering week to spend extending the lab, here's the
order that maximizes procurement-deck value per day:

| Day | Add | Why this slot |
|---|---|---|
| 1 | TimesFM | Most-cited; eliminates the most-asked-about gap |
| 2 | TTM | One day; adds the workstation-tier story |
| 3–4 | TimeMoE | Two days; pairs with existing `SubsystemMoE` for a head-to-head slide |
| 5 (morning) | Granite-Code config + procedure-synth A/B test | One hour code, half day to gather A/B numbers |
| 5 (afternoon) | Phi-4 configs + dual-use figure update | One hour code, half day for the figure refresh |

SAE work is a **separate research sprint** — flag it as a Phase 12
candidate rather than slotting it into the foundation-model extension
week. The deliverable (interpretable features) is high-impact but the
iteration cycle is genuinely uncertain.

---

## What's still NOT on this roadmap (and why)

| Model | Why skipped |
|---|---|
| TimeGPT (Nixtla) | Proprietary API; can't sneakernet |
| Time-LLM (Monash) | Research curiosity; LLM-as-TS pipeline is exotic and the value-add over native TS models has been disputed in 2024 follow-ups |
| GPT4TS / LLM4TS | Largely deprecated; field moved on |
| UniTS | Architecturally interesting but coverage overlap with TimesFM is high; pick one |
| Toto (Datadog) | Open-weights status was still evolving as of writing — revisit when licensing is unambiguous |
| Perceiver IO | Could replace `SharedCrossModalStack` but our current 3-modality version works fine; not worth rewriting until 5+ modalities |
| CodeBERT / BERT for command sequences | Pretrained on natural language; doesn't transfer to opcode tokens. Our from-scratch `SequenceTransformer` is the right call here. |
| Prophet (Meta) — for AD on raw telemetry | Stan/cmdstanpy dependency hostile to air-gap, univariate-only, training-time outlier removal silently drops the events we want to detect, and ETS covers the same algorithmic family with no extra deps. See [the deferral note](#why-prophet-was-deferred) for the full reasoning. Reconsider for a *separate* long-horizon-prognostics demo (battery degradation, fuel projection); see [Where Prophet might actually fit](#where-prophet-might-actually-fit-on-real-subsystems). |

---

## Cross-references

- Algorithm inventory (current state): [`packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md)
- Existing foundation-model adapters: [`packages/dgx_ts_lab/src/dgx_ts_lab/models/foundation/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/models/foundation/README.md)
- LLM co-pilot architecture: [`packages/dgx_ts_lab/src/dgx_ts_lab/llm/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/llm/README.md)
- Phase 7 explanation layer (where SAEs would attach): [`packages/dgx_ts_lab/src/dgx_ts_lab/explanation/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/explanation/README.md)
- Adding a model (general walkthrough): [`adding_a_model.md`](adding_a_model.md)
- Foundation-model weight provisioning (existing process): [`foundation_model_provisioning.md`](foundation_model_provisioning.md)
