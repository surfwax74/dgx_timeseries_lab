# Intern Onboarding — A Walkthrough of the Lab

A hands-on path from "I cloned the repo" to "I designed and ran my own
bake-off." Read each section, run the commands, look at the outputs,
do the exercises. By the end you'll have run experiments at every
**model level** in the lab — from a classical 50-line baseline through
to a full DGX-scale demo.

Plan for **~8 working hours** total, spread across as many days as you
like. Each section ends with a **checkpoint** — produce the artifact, save
it to your scratch folder, show your mentor.

---

## 0. Setup (~15 min, once)

### What this lab is

A satellite-telemetry **anomaly detection** research platform with
hot-swappable models, datasets, and trainers. The point is to run many
algorithms against the same data and produce comparable results.

### Environment check

```powershell
cd C:\dev\dgx_timeseries_lab

# Confirm Python, torch, and CUDA (or CPU) status
.\.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

# Set this once per shell so uv doesn't revert CUDA torch on you
$env:UV_NO_SYNC = '1'

# Smoke that the test suite passes
.\.venv\Scripts\python.exe -m pytest packages/ -q --tb=line
```

You should see `246 passed, 1 skipped` (the skipped one needs an
`ANTHROPIC_API_KEY` and is expected). If anything is broken, **stop and
ask your mentor** before continuing.

### Repo tour (5 min — open these in your editor and skim)

| File | What's in it |
|---|---|
| [`README.md`](../README.md) | Repo overview, the three Protocols, phase plan |
| [`docs/experiments_cookbook.md`](experiments_cookbook.md) | "How do I run X?" — bookmark this |
| [`packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md) | Algorithm inventory — every detector + algorithm family |
| [`docs/foundation_model_roadmap.md`](foundation_model_roadmap.md) | What's planned next + what we won't add |
| `configs/experiment/*.yaml` | All experiments — these are what you'll modify |
| `configs/model/*.yaml` | All model variants (tiny/small/medium/large for most) |
| `configs/trainer/*.yaml` | Per-tier trainer presets (cpu / rtx3080 / a5000 / h200 / etc.) |

### Make a scratch folder for your own runs

```powershell
mkdir intern_work\benchmark_reports
mkdir intern_work\notes
```

Save every report + figure + observation into `intern_work/`. At the end
of the week, this folder is what you present.

### Checkpoint 0

- [ ] Test suite passes
- [ ] You can open and read each of the 4 reference files above
- [ ] `intern_work/` folder exists

---

## 1. Level 1 — Classical baseline (~30 min)

**Goal**: Run the simplest possible bake-off and understand the
end-to-end output shape. No GPU needed.

### Step 1.1 — Run the quickstart

This runs three detectors (rolling-mean baseline + two small neural
detectors) on a trivial synthetic dataset and renders ROC/PR/AUC
figures. ~1 minute on CPU.

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main benchmark experiment=quickstart_viz
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main viz `
    --benchmark-dir benchmark_reports\quickstart_viz `
    --format png,svg --splits val,test
```

### Step 1.2 — Inspect what landed

Open these in order:

1. `benchmark_reports/quickstart_viz/benchmark_report.md` — the
   leaderboard. Note the ranking by `val ROC-AUC`.
2. `benchmark_reports/quickstart_viz/figures/auc_bar_val.png` — same
   numbers, visualized.
3. `benchmark_reports/quickstart_viz/figures/roc__trivial_synth__val.png`
   — the ROC curves. Find the chance diagonal. Find the best detector.
4. `benchmark_reports/quickstart_viz/figures/pr__trivial_synth__val.png`
   — the PR curves. **Why are they so different from the ROC curves?**
   (Hint: anomalies are rare — only 1% of timesteps. PR cares about that.)

### Step 1.3 — Look at one raw output file

```powershell
.\.venv\Scripts\python.exe -c @"
import numpy as np
data = np.load('benchmark_reports/quickstart_viz/rolling_mean__trivial_synth__s0__val.npz')
print('keys:', list(data.keys()))
print('scores shape:', data['scores'].shape)
print('labels shape:', data['labels'].shape)
print('first 10 scores:', data['scores'][:10])
print('first 10 labels:', data['labels'][:10])
print('anomaly fraction:', float(data['labels'].mean()))
"@
```

This is what every detector produces and what every viz function
consumes — two parallel arrays of (`scores[t]`, `labels[t]`).

### Exercise 1 — change a parameter

Edit `configs/experiment/quickstart_viz.yaml` and change one thing.
Some good first choices:

- `trainer.max_epochs: 1` → `3` (more training)
- `seeds: [0, 1]` → `seeds: [0, 1, 2, 3, 4]` (tighter error bars)
- `anomaly_rate: 0.02` → `0.05` (more anomalies — easier task)

Re-run the same two commands. Compare to your first results. **Save
both leaderboards to `intern_work/notes/level1_compare.md`** with a
one-paragraph observation about what changed.

### Checkpoint 1

- [ ] You can describe what `benchmark_report.md` contains
- [ ] You ran two variants and compared them
- [ ] Notes saved to `intern_work/notes/level1_compare.md`

---

## 2. Level 2 — From-scratch transformers (~1 hour)

**Goal**: Run the three transformer-based AD detectors (PatchTST+MAE,
AnomalyTransformer, DCdetector) on a more realistic dataset. CPU OK; a
3080 makes it noticeably faster.

### Step 2.1 — Run the workstation bake-off

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main benchmark experiment=rtx3080_bakeoff trainer=cpu trainer.max_epochs=2
```

We override `trainer=cpu` and shorten `max_epochs` so it finishes in
~5 min on CPU. On a 3080 you can drop `trainer=cpu` and let the
default trainer take over (~25 min, real epochs).

### Step 2.2 — Inspect the layered synthetic dataset

The previous bake-off used `trivial_synth` (sine + spikes). This one
uses `layered_synth` — a much richer dataset with orbital sinusoids,
mode transitions, drift, and 5 fault types. Run this to see the
fault catalog:

```powershell
.\.venv\Scripts\python.exe -c @"
from dgx_ts_lab.datasets.synthetic.layered import LayeredSyntheticDataset, faults, modes, noise, physics
from dgx_ts_core.data import Channel, Units, Subsystem
ds = LayeredSyntheticDataset(
    channels=[Channel(name='v', units=Units.VOLT, subsystem=Subsystem.EPS, sample_rate_hz=1.0)],
    components=[
        physics.OrbitalSinusoid('v', amplitude=1.0, period_s=5400.0),
        noise.GaussianNoise('v', std=0.05),
        faults.PointFault('v', rate_per_hour=10.0, magnitude=5.0),
    ],
    n_samples=3600, sample_rate_hz=1.0, seed=0,
)
print('Fault log entries:', len(ds._fault_log))
for f in ds._fault_log[:5]:
    print(' ', f)
print('Anomaly fraction:', float(ds._labels.mean()))
"@
```

That `fault_log` is what the labels come from. Each fault is tagged by
type, channel, time window, and severity. The 6 fault types
(`PointFault`, `DriftFault`, `StuckAtFault`, `DropoutFault`,
`OscillationFault`, `CorrelationBreakFault`) are why neural detectors
can beat rolling-mean — the variety overwhelms a simple z-score.

### Step 2.3 — Render the figures

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main viz `
    --benchmark-dir benchmark_reports\rtx3080_bakeoff `
    --format png --splits val,test
```

Open `benchmark_reports/rtx3080_bakeoff/figures/auc_bar_val.png`. With
2 CPU epochs the neural detectors are under-trained and rolling-mean
likely still wins — that's expected. **Save a screenshot to
`intern_work/notes/level2_undertrained.png`.**

### Exercise 2 — copy the config and tune it

Copy the experiment to your own file:

```powershell
copy configs\experiment\rtx3080_bakeoff.yaml configs\experiment\intern_2a_bigger.yaml
```

Edit `configs/experiment/intern_2a_bigger.yaml`:

- Bump `trainer.max_epochs` to **20**
- Bump `trainer.batch_size` to **64**
- Bump the per-detector `d_model` from 128 to **256**
- Bump `n_layers` from 3 to **4**
- Change `suite.name` to `intern_2a_bigger` so output goes to a
  different folder

Run it:

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main benchmark experiment=intern_2a_bigger
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main viz --benchmark-dir benchmark_reports\intern_2a_bigger
```

This takes ~10 min on a 3080 or ~40 min on CPU. **Save the new figures
side-by-side with the under-trained run in
`intern_work/notes/level2_compare.md`** with notes about which detector
benefited most from the longer training.

### Checkpoint 2

- [ ] Both bake-offs completed
- [ ] You can name the 3 neural detectors and have a one-line opinion
      on each
- [ ] `intern_2a_bigger.yaml` is your first authored experiment

---

## 3. Level 3 — Foundation models (~1 hour)

**Goal**: Add zero-shot pretrained foundation models (TimesFM, TTM,
TimeMoE) to a bake-off. No fine-tuning, just calibrate the threshold.

### Step 3.1 — Run the TSFM smoke

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main benchmark experiment=tsfm_smoke
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main viz --benchmark-dir benchmark_reports\tsfm_smoke
```

You should see all 4 detectors complete (rolling_mean + 3 TSFMs). The
TSFMs are running their **untrained fallback** path because no Hugging
Face weights are present on this box — they fall back to small
randomly-initialized models. AUCs will be weak (~0.35–0.55); that's
expected without weights.

> **Important**: this run proves the plumbing works. For real numbers
> you need the actual HF weights sneakernet'd to the box. See
> `docs/foundation_model_roadmap.md` § "Air-gap weight provisioning
> workflow" for the bundle process.

### Step 3.2 — Read the model inventory

Open
[`packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md).

Find the **at-a-glance** table. Pick **three** detectors you haven't run
yet from different family rows (e.g. `chronos`, `subsystem_moe`,
`sat_tsfm`). Write a one-line description of each into
`intern_work/notes/level3_picks.md`.

### Exercise 3 — bring your own bake-off

Copy `tsfm_smoke.yaml` to your own:

```powershell
copy configs\experiment\tsfm_smoke.yaml configs\experiment\intern_3a_mine.yaml
```

In `configs/experiment/intern_3a_mine.yaml`:

- Add the three detectors from your picks above to `suite.detectors`
  (look at how the existing rows are formatted — the registry keys are
  the strings you put in `key:`)
- Change `suite.name` to `intern_3a_mine`
- Look up each detector's required params in `configs/model/<key>.yaml`
  (or the existing experiments that already use them)

Run + viz:

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main benchmark experiment=intern_3a_mine
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main viz --benchmark-dir benchmark_reports\intern_3a_mine
```

If you get `'X' not registered. Available: [...]`, copy a working `key`
from one of the existing experiment YAMLs and adapt.

### Checkpoint 3

- [ ] You ran 3 detectors you'd never run before
- [ ] Notes on each detector saved
- [ ] One observation about *why* foundation models trail rolling-mean
      in the untrained-fallback regime (this is a pretty good interview
      answer — work on it)

---

## 4. Level 4 — Real data: NASA SMAP / MSL (~1 hour)

**Goal**: Move from synthetic to real spacecraft telemetry.

### Step 4.1 — Get the data

If `data/nasa_smap/` and `data/nasa_msl/` don't exist yet on this box,
follow [`docs/experiments_cookbook.md`](experiments_cookbook.md)
§ "Phase 1 — Layered synthetic + NASA loaders" — the steps cover the
download, the per-spacecraft split, and the verification probe. The
full procedure was also documented in the chat where the lab was built;
ask your mentor for it if the docs are unclear.

### Step 4.2 — Verify the loader can read it

```powershell
.\.venv\Scripts\python.exe -c @"
from dgx_ts_lab.datasets.nasa_telemanom import NasaTelemanomChannel
ds = NasaTelemanomChannel(data_root='data/nasa_smap', channel_id='A-1', spacecraft='SMAP')
print('Loaded:', ds.name, '| samples:', len(ds._data), '| anomaly fraction:', float(ds._labels.mean()))
"@
```

If that prints sensible numbers, you're set.

### Step 4.3 — Run one detector on one real channel

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main train `
    dataset=smap dataset.channel_id=A-1 `
    model=patchtst_mae trainer=cpu trainer.max_epochs=5
```

You'll get an MLflow run + a checkpoint + a per-step score trace
against the held-out test split. Look for `val ROC-AUC` in the console
output.

### Exercise 4 — build a multi-channel NASA bake-off

Create `configs/experiment/intern_4a_nasa.yaml`:

```yaml
# @package _global_
defaults:
  - override /trainer: cpu

trainer:
  max_epochs: 5
  batch_size: 32

suite:
  name: intern_4a_nasa
  mode: pretrain

  detectors:
    - {key: rolling_mean,        params: {}}
    - {key: patchtst_mae,        params: {window_length: 128, patch_len: 16, d_model: 64, n_layers: 2, n_heads: 2}}
    - {key: anomaly_transformer, params: {window_length: 128, d_model: 64, n_layers: 2, n_heads: 2}}

  datasets:
    - {key: nasa_smap_channel, params: {data_root: data/nasa_smap, channel_id: A-1, spacecraft: SMAP}}
    - {key: nasa_smap_channel, params: {data_root: data/nasa_smap, channel_id: P-1, spacecraft: SMAP}}
    - {key: nasa_smap_channel, params: {data_root: data/nasa_smap, channel_id: T-1, spacecraft: SMAP}}

  seeds: [0, 1]

output_dir: benchmark_reports
```

Then:

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main benchmark experiment=intern_4a_nasa
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main viz --benchmark-dir benchmark_reports\intern_4a_nasa
```

### Checkpoint 4

- [ ] You loaded a real NASA channel
- [ ] You ran a multi-channel × multi-detector × multi-seed sweep on
      real telemetry
- [ ] You can articulate which detector won on which channel and venture
      a guess as to why

---

## 5. Level 5 — LoRA fine-tuning (~30 min on a GPU, skippable on CPU)

**Goal**: Take a pretrained foundation model and adapt it to your
dataset with LoRA. Needs a GPU and real HF weights to be meaningful —
if both are absent, skim this section and skip to Level 6.

### Step 5.1 — Look at the diff between `_zero` and `_lora` configs

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; print(open('configs/model/chronos_zero.yaml').read()); print('---'); print(open('configs/model/chronos_lora.yaml').read())"
```

The difference is small — the `_lora` variant uses the same backbone
but during fit the trainer wraps it in LoRA adapters. The PEFT config
lives in the trainer (`configs/trainer/*.yaml` under `extra.peft`).

### Step 5.2 — Run a LoRA fine-tune on your hardware tier

```powershell
# On A5000 / H200:
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main train `
    experiment=phase3_bakeoff trainer=a5000 `
    "suite.detectors=[{key: chronos_lora, params: {model: amazon/chronos-t5-tiny}}]"
```

### Exercise 5 — compare zero-shot vs LoRA

Build `intern_5a_lora_vs_zero.yaml` that runs the same Chronos model in
both modes against the layered synth. Submit a leaderboard delta.

### Checkpoint 5

- [ ] You understand what LoRA does and where the adapter weights live
- [ ] (If you had a GPU + weights) You produced one comparison run

---

## 6. Level 6 — Specialized model families (~1 hour)

**Goal**: Run the three "non-AD" model families: multi-task heads
(Phase 6), physics-informed (Phase 9), and cyber/behavior (Phase 8).

### Step 6.1 — Multi-task heads

The Sat-TSFM multi-task model trains one encoder + four task heads
simultaneously (AD + fault classification + RUL regression + mode
prediction). Run:

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main train experiment=phase6_multitask
```

Watch the per-task metrics scroll by. **What does `rul_regressor.mae_log_s`
mean? Why log seconds?** (Open `packages/dgx_ts_lab/src/dgx_ts_lab/models/heads/README.md`
for the answer.)

### Step 6.2 — Physics-informed AD

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main train experiment=phase9_pinn_bakeoff
```

This runs a thermal PINN + ADCS PINN against synthetic data with known
physics violations. Look at how the loss splits into `data_loss` +
`physics_loss` in the MLflow output.

### Step 6.3 — Cyber AD

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main train experiment=phase8_cyber
```

This runs the SequenceTransformer (BERT-MLM over command tokens) +
OperatorFingerprint (Mahalanobis distance) against synthetic command
sequences with priv-escalation injections.

### Exercise 6 — design a 4-detector ladder

Build `intern_6a_ladder.yaml` that runs **one** representative from
each tier on the layered synth:

- `rolling_mean` (classical)
- `patchtst_mae` (from-scratch transformer)
- `chronos_zero` (pretrained foundation)
- `sat_tsfm_multitask` (specialized)

The point: show the four model levels side-by-side on the same data.
This is exactly the procurement-deck argument.

### Checkpoint 6

- [ ] You ran one example from each specialized family
- [ ] You produced a 4-detector ladder run
- [ ] You can sketch the model-family taxonomy without looking it up

---

## 7. Level 7 — The DGX showcase (read-only, ~30 min)

**You will not run this** unless you have DGX time booked. But read
through it so you know what the lab can do at the top.

### Step 7.1 — Read the docs

- [`docs/foundation_model_roadmap.md`](foundation_model_roadmap.md)
- [`docs/llm_ops_copilot.md`](llm_ops_copilot.md)
- `scripts/dgx_showcase.sh` (the orchestration script)
- `configs/experiment/dgx_showcase.yaml` (Sat-TSFM XL multi-task + FSDP-8)
- `configs/experiment/dgx_showcase_multimodal.yaml` (cross-modal foundation)

### Step 7.2 — Look at the figures

Open `benchmark_reports/capability_cliff/*.png` — these are the four
procurement-deck figures (capability ladder, capability matrix, DGX vs
federated, dual-use capacity). Understand each one well enough to
narrate it.

### Checkpoint 7

- [ ] You can explain what `dgx_showcase.sh` does in 30 seconds
- [ ] You can articulate why an 8x H200 DGX is qualitatively different
      from 8 federated PCIe GPUs (hint: NVSwitch)

---

## 8. Graduation — design your own bake-off (2 hours)

**Goal**: Pick a research question, build a config, run, analyze,
present.

### Step 8.1 — Pick a question

Examples (steal one or invent your own):

- "Does the workstation tier need foundation models, or is PatchTST+MAE
  enough?" → Compare PatchTST+MAE vs Chronos / TimesFM zero-shot on
  NASA SMAP across 5 channels, 3 seeds each.
- "Does increasing context window matter more than model size?" →
  PatchTST+MAE at 4 (window × d_model) combinations on the LEO EPS
  synth.
- "How robust are detectors to noise type?" → One detector, the
  layered synth, sweeping the noise components (Gaussian, pink,
  Student-t, Poisson burst).
- "Which is more robust to anomaly-class imbalance?" → A bake-off where
  you vary `anomaly_rate` from 0.01 to 0.20 in 5 steps.

Save your question to `intern_work/notes/graduation_question.md`.

### Step 8.2 — Build the config

Copy the closest existing experiment, adapt, save as
`configs/experiment/intern_graduation.yaml`. The cookbook's "Hydra
overrides cheat-sheet" section is your friend.

### Step 8.3 — Run

```powershell
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main benchmark experiment=intern_graduation
.\.venv\Scripts\python.exe -m dgx_ts_lab.cli.main viz --benchmark-dir benchmark_reports\intern_graduation --format png,svg
```

### Step 8.4 — Write up

Produce `intern_work/graduation_report.md` with:

1. The question
2. The experimental design (what you swept, why)
3. The leaderboard (paste the table)
4. The figures (link or embed)
5. **Three observations** — one per significant numerical finding
6. **One follow-up question** that would justify a second experiment

### Final checkpoint

- [ ] You ran an experiment nobody told you exactly what to run
- [ ] Your graduation_report.md exists, is honest about what worked
      and what didn't, and would be presentable to a research review

---

## When you get stuck

| Symptom | Look at |
|---|---|
| "Could not find experiment/X" | Wrong CLI selector — see the cookbook § "Spotting CLI/group mismatches" |
| "Detector not registered" | `models/from_scratch/__init__.py` — did the import side-effect fire? |
| Tests fail after I edited something | Run `pytest packages/<pkg>/tests/test_*.py -x` to find the first failure |
| Plots don't render | Did the benchmark write `*.npz` files? Without them viz has no data |
| `uv sync` reverted CUDA torch | `$env:UV_NO_SYNC = '1'` then re-run |
| Anything else | Check the cookbook's "Where to look when something breaks" table |

## When you're really stuck

Open a one-page issue draft for your mentor:

```markdown
**What I'm trying to do**:
**Exact command I ran**:
**Full error message**:
**What I've already tried**:
```

Don't paste 500-line tracebacks — capture the first error line + the
exception type + the file:line it surfaced at. Half the time writing
this draft makes you find the answer yourself.

---

## What you should know by the end

If you can answer all of these from memory, you've completed the
onboarding:

1. What does `Capabilities` declare and why is it important for hot-swap?
2. What does `_target_key` do in a Hydra YAML?
3. Why does `dgx-ts synth` take `dataset=` but `dgx-ts train` take
   `experiment=`?
4. What's the difference between AD, forecasting, and RUL — and which
   models in the lab address each?
5. Why does rolling_mean beat untrained foundation models on
   trivial_synth but not on layered_synth with real weights?
6. Where do the per-run `.npz` files come from and what does `dgx-ts viz`
   do with them?
7. What is the procurement argument for a dedicated DGX vs. 8 federated
   PCIe GPUs in one sentence?

Bring this list to your wrap-up meeting. Your mentor will pick three
at random.

---

## Where to go next

- **Forecasting + RUL bake-off** — scoped in
  [`docs/forecasting_rul_bakeoff.md`](forecasting_rul_bakeoff.md). Five
  task units mapped out, suitable for a second internship rotation.
- **New foundation model adapter** — TimesFM/TTM/TimeMoE landed
  recently; if you want to add a fourth, the
  [`docs/foundation_model_roadmap.md`](foundation_model_roadmap.md)
  has the "Standard add-a-foundation-model pattern" recipe.
- **Cyber dataset extension** — add a new attack class to
  `datasets/synthetic/cyber/command_sequence_gen.py` and demonstrate
  whether the existing SequenceTransformer catches it.
- **Phase 7 explanation report** — wire `dgx-ts explain` against one
  of your runs and read the auto-generated Markdown.
