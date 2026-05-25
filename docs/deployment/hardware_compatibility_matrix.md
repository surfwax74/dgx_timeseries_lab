# Hardware compatibility matrix

What model × tier combinations actually fit and complete in reasonable wall time? Use this table to pick experiments before kicking off a long run.

Notation: ✓ fits comfortably / ⚠ tight / ✗ won't fit (or impractical wall time).

## Per-model fit by tier

| Model (key) | Approx params | CPU | 1× RTX 3080 (12 GB) | 2× RTX 3080 (24 GB) | 1× A5000 (24 GB) | 8× A5000 (192 GB) | 1× H200 (141 GB) | 8× H200 (1.1 TB) |
|---|---:|---|---|---|---|---|---|---|
| `rolling_mean` | 0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `anomaly_transformer` (tiny: 1M) | 1 M | ⚠ slow | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `patchtst_mae` (tiny) | 1 M | ⚠ slow | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `dcdetector` (tiny) | 0.5 M | ⚠ slow | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `anomaly_transformer` (small) | 10 M | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `patchtst_mae` (base) | 30 M | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `chronos_zero` (tiny T5: 8M) | 8 M | ⚠ slow | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `chronos_lora` (small: 60M) | 60 M | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `chronos_full_finetune` (base: 200M) | 200 M | ✗ | ✗ | ✓ DDP | ✓ | ✓ | ✓ | ✓ |
| `moment_lora` (base: 110M) | 110 M | ✗ | ⚠ tight | ✓ DDP | ✓ | ✓ | ✓ | ✓ |
| `moment_full_finetune` (large: 350M) | 350 M | ✗ | ✗ | ✗ | ⚠ tight | ✓ FSDP | ✓ | ✓ |
| `moirai_lora` (small) | 14 M | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `moirai_lora` (large: 311M) | 311 M | ✗ | ✗ | ⚠ tight | ✓ | ✓ FSDP | ✓ | ✓ |
| **Sat-TSFM small** *(Phase 4)* | 50 M | ✗ | ✓ tight | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Sat-TSFM medium** *(Phase 4)* | 200 M | ✗ | ✗ | ✗ | ⚠ tight | ✓ DDP | ✓ | ✓ |
| **Sat-TSFM large** *(Phase 4)* | 500 M | ✗ | ✗ | ✗ | ✗ | ✓ FSDP | ✓ | ✓ |
| **Sat-TSFM xlarge** *(Phase 4)* | 1 B | ✗ | ✗ | ✗ | ✗ | ⚠ tight FSDP | ⚠ tight | ✓ FSDP |
| **Sat-TSFM huge** *(Phase 4)* | 3 B+ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ FSDP + act-ckpt |

## Per-dataset wall time at each tier

Estimates for one full training pass (Phase 2 bake-off — 4 detectors × N epochs):

| Dataset | CPU | RTX 3080 | A5000 | 8× A5000 | H200 | 8× H200 |
|---|---|---|---|---|---|---|
| `trivial_synth` (5k × 3 ch) | < 1 min | < 30 s | < 10 s | < 10 s | < 5 s | < 5 s |
| `presets/leo_eps_24h` (86k × 6 ch) | ~5 min | ~30 s | ~10 s | ~5 s | < 5 s | < 5 s |
| `presets/leo_eps_full_24h` (86k × 83 ch) | ~30 min | ~3 min | ~1 min | ~30 s | ~20 s | ~10 s |
| NASA SMAP (single channel) | ~2 min | ~30 s | ~10 s | ~5 s | < 5 s | < 5 s |
| Multi-day synth (1M × 83 ch) | ✗ impractical | ~30 min | ~5 min | ~2 min | ~1 min | ~30 s |

## Recommended starting points by tier

| Tier | First experiment | Stretch experiment |
|---|---|---|
| **CPU** | `phase0_smoke` | `phase1_layered` |
| **RTX 3080** | `phase2_bakeoff` × `presets/leo_eps_full_24h` | `phase3_bakeoff` with chronos-tiny LoRA |
| **A5000 ×1** | Full Phase 2 with reasonable epoch counts (50+) | Phase 3 with chronos-small/moment-base LoRA |
| **A5000 ×8** | Phase 3 bake-off, including full fine-tunes | Phase 4 Sat-TSFM medium DDP |
| **H200 ×1** | Phase 4 Sat-TSFM medium training | Long-context (4K) ablations |
| **DGX 8×H200** | Phase 4 Sat-TSFM large FSDP | Phase 4 Sat-TSFM 1B+ multi-day run |

## Rules of thumb for picking model size

1. **Fits comfortably** (model ≤ 30% of available VRAM): use DDP if multi-GPU, otherwise single-device.
2. **Fits but tight** (model 30–70% VRAM): bf16-mixed mandatory, reduce batch size, consider FSDP.
3. **Doesn't fit on one GPU** (model > VRAM): FSDP required, +activation checkpointing, +grad accumulation.

## When to scale up

- **Iterating on architecture / dataset / config**: stay on the smallest tier that gives you signal in < 30 min per run.
- **Producing benchmark numbers for a paper / report**: jump to the largest tier you have available; epoch counts and model sizes need to be at "real" scale.
- **Demonstrating the platform to a stakeholder**: DGX 8×H200 with Sat-TSFM 1B, FSDP, long context — that's the headline.
