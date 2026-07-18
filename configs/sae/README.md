# configs/sae/

Sparse Autoencoder configs for the interpretability sprint. See the
code + design notes at
[`packages/dgx_ts_lab/src/dgx_ts_lab/explanation/sae/README.md`](../../packages/dgx_ts_lab/src/dgx_ts_lab/explanation/sae/README.md).

## Files

| YAML | Target model | d_input | d_dict | k | Tier |
|---|---|---:|---:|---:|---|
| `topk_sae_small.yaml` | Sat-TSFM small | 256 | 2048 (8x) | 32 | CPU / RTX 3080 |
| `topk_sae_medium.yaml` | Sat-TSFM medium | 512 | 8192 (16x) | 64 | A5000 / H200 |

## Sizing guidance

* `d_dict` should be 8x to 32x `d_input`. Wider = more interpretable
  features but more compute.
* `k` should be a small multiple of `sqrt(d_dict)`. Sparser = more
  interpretable per-atom but higher reconstruction loss.
* `aux_loss_weight = 1/32` is the Gao et al. sweet spot for dead-neuron
  revival — bump it if `dead_atom_fraction` stays >5% at end of training.
