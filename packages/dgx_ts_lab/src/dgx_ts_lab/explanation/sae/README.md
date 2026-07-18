# Sparse Autoencoders (SAE) — interpretability sprint

A Sparse Autoencoder learns an over-complete dictionary of features
whose activations are enforced to be sparse (only the top-k largest
activations per input are kept). Trained on frozen encoder activations
from Sat-TSFM (or any other model), the dictionary atoms become
human-interpretable features — you can name each atom by looking at
which input windows activate it maximally.

**Purpose in this repo**: give operators an interpretable dictionary of
what Sat-TSFM "thinks about" — specific failure modes, orbital phases,
subsystem-coupling patterns — instead of the opaque `d_model`-dim
encoder output.

## What's here (Wave 1 — this sprint)

* `sae.py` — `TopKSAE` model (Gao et al. centered-SAE variant with tied
  weights, ReLU top-k activation, dead-neuron aux loss).
* `train.py` — `train_sae()` fitting loop that consumes a fixed batch of
  pre-captured activations. History dataclass includes per-epoch
  reconstruction loss + dead-atom fraction.
* `__init__.py` — exports `TopKSAE` and `train_sae`.

## What's NOT here yet (Wave 2)

* **Activation capture** — hooks that freeze a Sat-TSFM checkpoint,
  stream data through, and dump `(N, d_model)` activations to parquet.
  This is the natural next sprint; it needs design decisions about
  which layer(s) to hook and whether to capture pre- or post-LayerNorm.
* **Feature interpretation** — for each dictionary atom, find the top-K
  input windows that activate it most. This is a search over the
  training corpus; will live alongside the AD explanation module.
* **Integration with `dgx-ts explain`** — currently the explanation
  layer uses gradient/attribution methods; SAE dictionaries slot in as
  a complementary "what is this run doing" view.

## Quick usage

```python
import numpy as np
from dgx_ts_lab.explanation.sae import TopKSAE, train_sae
from dgx_ts_lab.explanation.sae.train import SAETrainingConfig

# Pretend `acts` is (N, 256) captured from a frozen Sat-TSFM encoder.
acts = np.random.randn(10_000, 256).astype("float32")

sae = TopKSAE(d_input=256, d_dict=2048, k=32)
history = train_sae(
    sae,
    activations=acts,
    config=SAETrainingConfig(n_epochs=20, batch_size=256),
    device="cpu",
)
print("final recon loss:", history.recon_loss[-1])
print("dead atom fraction:", history.dead_atom_fraction[-1])
```

## Design references

* Cunningham et al., "Sparse Autoencoders Find Highly Interpretable
  Features in Language Models" (2023).
* Gao, Goh, Sutskever et al. (OpenAI), "Scaling and Evaluating Sparse
  Autoencoders" (2024) — the centered-SAE + dead-neuron revival tricks
  are from this paper.
* Bricken et al. (Anthropic), "Towards Monosemanticity" (2023) — the
  broader framing of what SAE features look like in practice.

## Config

The default hyperparameters in `SAETrainingConfig` are calibrated for
a small SAE (d_input <= 256, d_dict <= 4096) on CPU. For DGX-scale
runs, see `configs/sae/topk_sae_medium.yaml`.
