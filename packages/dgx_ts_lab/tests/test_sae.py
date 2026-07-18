"""TopKSAE + train_sae unit tests.

Covers:
  * Encode / decode shapes.
  * Top-k sparsity is exactly k.
  * Reconstruction on trivial random data drops with training.
  * Firing-count and dead-atom bookkeeping.
  * Auxiliary loss returns 0 when not enough dead atoms exist.
  * train_sae runs end-to-end and returns a well-formed history.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dgx_ts_lab.explanation.sae import TopKSAE, train_sae
from dgx_ts_lab.explanation.sae.train import SAETrainingConfig


def test_encode_returns_topk_values_and_indices() -> None:
    sae = TopKSAE(d_input=8, d_dict=32, k=4)
    x = torch.randn(5, 8)
    values, indices = sae.encode(x)
    assert values.shape == (5, 4)
    assert indices.shape == (5, 4)
    # Values must be non-negative (post-ReLU).
    assert torch.all(values >= 0)
    # Indices must be in [0, d_dict).
    assert torch.all(indices >= 0)
    assert torch.all(indices < sae.d_dict)


def test_encode_dense_has_exact_topk_sparsity() -> None:
    sae = TopKSAE(d_input=16, d_dict=64, k=8)
    x = torch.randn(10, 16)
    z = sae.encode_dense(x)
    assert z.shape == (10, 64)
    # Each row should have EXACTLY k non-zero entries (or fewer if some
    # top-k values are exactly zero — unlikely but count non-zero directly).
    n_nonzero = (z != 0).sum(dim=-1)
    assert torch.all(n_nonzero <= 8)   # at most k
    # In expectation, with random inputs, virtually all should be exactly k.
    assert (n_nonzero == 8).float().mean() > 0.9


def test_decode_reconstructs_from_encode_output() -> None:
    sae = TopKSAE(d_input=4, d_dict=16, k=2)
    x = torch.randn(3, 4)
    values, indices = sae.encode(x)
    recon = sae.decode(values, indices)
    assert recon.shape == x.shape


def test_forward_returns_tuple_of_three() -> None:
    sae = TopKSAE(d_input=4, d_dict=16, k=2)
    x = torch.randn(3, 4)
    recon, values, indices = sae(x)
    assert recon.shape == (3, 4)
    assert values.shape == (3, 2)
    assert indices.shape == (3, 2)


def test_firing_stats_increment_on_forward() -> None:
    sae = TopKSAE(d_input=4, d_dict=16, k=2)
    x = torch.randn(5, 4)
    initial = sae.firing_count.sum().item()
    _ = sae(x)
    # 5 rows * k=2 atoms = 10 total firings.
    assert sae.firing_count.sum().item() - initial == 10


def test_reset_firing_stats_zeroes_counts() -> None:
    sae = TopKSAE(d_input=4, d_dict=16, k=2)
    x = torch.randn(5, 4)
    _ = sae(x)
    sae.reset_firing_stats()
    assert sae.firing_count.sum().item() == 0


def test_dead_atom_fraction_bounds() -> None:
    sae = TopKSAE(d_input=4, d_dict=16, k=2)
    # Before any forward, ALL atoms are dead.
    assert sae.dead_atom_fraction() == 1.0
    x = torch.randn(100, 4)
    _ = sae(x)
    # After 100 samples * k=2 firings, at most 200 atom-activations across
    # 16 atoms — some will fire but not all.
    frac = sae.dead_atom_fraction()
    assert 0.0 <= frac <= 1.0


def test_aux_loss_zero_when_no_dead_atoms_available() -> None:
    """If fewer atoms are dead than we want to revive, aux_loss returns 0."""
    sae = TopKSAE(d_input=4, d_dict=8, k=8)  # k == d_dict => all atoms fire
    x = torch.randn(3, 4)
    recon, _, _ = sae(x)
    aux = sae.aux_loss(x, recon, n_dead_to_revive=4)
    assert aux.item() == 0.0


def test_ctor_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be in"):
        TopKSAE(d_input=4, d_dict=16, k=0)
    with pytest.raises(ValueError, match="k must be in"):
        TopKSAE(d_input=4, d_dict=16, k=17)


def test_untied_and_tied_variants_both_run() -> None:
    for untied in (False, True):
        sae = TopKSAE(d_input=4, d_dict=16, k=2, untie_weights=untied)
        x = torch.randn(3, 4)
        recon, _, _ = sae(x)
        assert recon.shape == x.shape


def test_train_sae_reduces_reconstruction_loss() -> None:
    """The recon loss on epoch N should be <= the loss on epoch 0."""
    torch.manual_seed(0)
    np.random.seed(0)

    # Synthetic activations from a low-rank linear combination — SHOULD
    # be recoverable by an over-complete dictionary.
    n_samples, d_input, latent = 1000, 32, 8
    rng = np.random.default_rng(0)
    W = rng.normal(size=(latent, d_input)).astype("float32")
    codes = rng.normal(size=(n_samples, latent)).astype("float32")
    acts = codes @ W

    sae = TopKSAE(d_input=d_input, d_dict=256, k=16)
    cfg = SAETrainingConfig(n_epochs=6, batch_size=128, learning_rate=1e-3)
    hist = train_sae(sae, acts, config=cfg)

    assert len(hist.recon_loss) == 6
    # First-epoch loss should exceed last-epoch loss with a meaningful margin.
    assert hist.recon_loss[0] > hist.recon_loss[-1] * 1.05


def test_train_sae_rejects_dim_mismatch() -> None:
    sae = TopKSAE(d_input=8, d_dict=32, k=4)
    with pytest.raises(ValueError, match="activations dim"):
        train_sae(sae, np.zeros((10, 16), dtype="float32"), config=SAETrainingConfig(n_epochs=1))


def test_train_sae_history_shape() -> None:
    sae = TopKSAE(d_input=4, d_dict=16, k=2)
    hist = train_sae(sae, np.random.randn(50, 4).astype("float32"),
                     config=SAETrainingConfig(n_epochs=3, batch_size=8))
    assert len(hist.epoch) == 3
    assert len(hist.recon_loss) == 3
    assert len(hist.aux_loss) == 3
    assert len(hist.dead_atom_fraction) == 3
