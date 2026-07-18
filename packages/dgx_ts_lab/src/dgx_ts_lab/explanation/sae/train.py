"""Minimal training loop for TopKSAE on pre-captured activations.

The typical flow:
  1. Freeze a Sat-TSFM (or other) encoder.
  2. Feed a corpus through it, saving the (N, d_model) activations to
     disk or holding them in memory.
  3. Call `train_sae(sae, activations, ...)` to fit a TopKSAE on top.

Activation capture lives in a separate module (planned as Wave 2 of
the SAE sprint) so this trainer stays decoupled from any particular
upstream encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .sae import TopKSAE


@dataclass
class SAETrainingConfig:
    """Hyperparameters for a single training run.

    Defaults chosen to work reasonably on a small (d_input <= 256,
    d_dict <= 4096) SAE running on CPU. For d_input >= 512, bump
    `batch_size` to 4096+ and put it on GPU.
    """

    n_epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 3e-4
    aux_loss_weight: float = 1 / 32
    dead_atom_reset_interval_epochs: int = 5
    normalize_decoder_after_step: bool = True
    log_every_epochs: int = 1


@dataclass
class SAETrainingHistory:
    """Per-epoch history returned by `train_sae`."""

    epoch: list[int] = field(default_factory=list)
    recon_loss: list[float] = field(default_factory=list)
    aux_loss: list[float] = field(default_factory=list)
    dead_atom_fraction: list[float] = field(default_factory=list)


def train_sae(
    sae: TopKSAE,
    activations: np.ndarray | torch.Tensor,
    config: SAETrainingConfig | None = None,
    device: str | torch.device = "cpu",
    on_epoch_end: Callable[[int, SAETrainingHistory], None] | None = None,
) -> SAETrainingHistory:
    """Train a TopKSAE on a fixed batch of pre-captured activations.

    Parameters
    ----------
    sae
        The `TopKSAE` to train (mutated in place).
    activations
        (N, d_input) array of encoder activations to fit the SAE on.
        Can be a numpy array or a torch tensor.
    config
        Training hyperparameters. Uses `SAETrainingConfig()` defaults if
        None.
    device
        Target device. `sae` is moved to `device` at the top of the call.
    on_epoch_end
        Optional callback for MLflow logging / early stopping. Called as
        `on_epoch_end(epoch, history)` after each epoch.
    """
    cfg = config or SAETrainingConfig()
    device = torch.device(device)
    sae = sae.to(device)

    if isinstance(activations, np.ndarray):
        activations = torch.from_numpy(activations)
    activations = activations.to(torch.float32).to(device)

    if activations.shape[1] != sae.d_input:
        raise ValueError(
            f"activations dim {activations.shape[1]} != sae.d_input {sae.d_input}"
        )

    dataset = TensorDataset(activations)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    optimizer = optim.Adam(sae.parameters(), lr=cfg.learning_rate)
    history = SAETrainingHistory()

    for epoch in range(cfg.n_epochs):
        if epoch % cfg.dead_atom_reset_interval_epochs == 0:
            sae.reset_firing_stats()

        recon_running = 0.0
        aux_running = 0.0
        n_batches = 0

        sae.train()
        for (x,) in loader:
            recon, _, _ = sae(x)
            l_recon = sae.reconstruction_loss(x, recon)
            l_aux = sae.aux_loss(x, recon)
            loss = l_recon + cfg.aux_loss_weight * l_aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if cfg.normalize_decoder_after_step:
                sae._normalize_decoder_rows()  # noqa: SLF001 - internal by design

            recon_running += float(l_recon.item())
            aux_running += float(l_aux.item())
            n_batches += 1

        history.epoch.append(epoch)
        history.recon_loss.append(recon_running / max(1, n_batches))
        history.aux_loss.append(aux_running / max(1, n_batches))
        history.dead_atom_fraction.append(sae.dead_atom_fraction())

        if on_epoch_end is not None:
            on_epoch_end(epoch, history)

    return history
