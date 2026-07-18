"""TopK Sparse Autoencoder — the core interpretability primitive.

Learns an over-complete dictionary D (of shape d_dict x d_input) whose
atoms serve as interpretable "features" of the input activations. The
top-k enforced sparsity keeps only the k largest-magnitude activations
per input, which:

  * eliminates the L1 penalty (top-k gives exact 0-vs-nonzero without
    needing a soft prior),
  * makes reconstruction loss the only training signal, and
  * gives interpretable, monosemantic features when the dictionary is
    wide enough (typically 8x-32x the input dimension).

References
----------
* Cunningham et al., "Sparse Autoencoders Find Highly Interpretable
  Features in Language Models", 2023.
* Gao, Goh, Sutskever et al. (OpenAI), "Scaling and Evaluating Sparse
  Autoencoders", 2024.

Design decisions
----------------
* **Tied weights** — decoder = encoder.T by default. This is Gao et al.'s
  finding: untied weights don't buy meaningful reconstruction quality but
  double the parameter count and can hurt feature interpretability.
  We keep an `untie_weights=True` escape hatch for research runs.
* **No bias in encoder** — decoder bias handles the mean offset; adding
  an encoder bias just shifts what "zero-activation" means without
  changing capacity. The `subtract_decoder_bias_before_encode` flag
  implements the Gao et al. "centered SAE" refinement.
* **Auxiliary dead-neuron loss** is exposed via the `aux_loss()` method
  but NOT combined into the primary loss automatically — the training
  loop is responsible for weighting it. This keeps the model class
  loss-agnostic.
"""

from __future__ import annotations

import torch
from torch import nn


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder.

    Parameters
    ----------
    d_input : int
        Dimensionality of the activations being encoded (e.g. Sat-TSFM
        encoder output d_model).
    d_dict : int
        Dictionary size. Typical: 8x to 32x d_input. Larger = more
        interpretable features but more compute.
    k : int
        Number of dictionary atoms allowed to activate per input.
        Rule of thumb: k = sqrt(d_dict) or a small multiple of it.
    untie_weights : bool
        If False (default), decoder weights = encoder.T so total params
        scale as (d_input + 1) * d_dict. If True, encoder and decoder
        are independent — twice as many params.
    subtract_decoder_bias_before_encode : bool
        Centered-SAE trick from Gao et al. — subtracting the decoder
        bias from the input before encoding decorrelates the "always-on"
        offset from the learned features.
    """

    def __init__(
        self,
        d_input: int,
        d_dict: int,
        k: int,
        untie_weights: bool = False,
        subtract_decoder_bias_before_encode: bool = True,
    ) -> None:
        super().__init__()
        if k <= 0 or k > d_dict:
            raise ValueError(f"k must be in [1, d_dict={d_dict}]; got {k}")
        self.d_input = d_input
        self.d_dict = d_dict
        self.k = k
        self.untie_weights = untie_weights
        self.subtract_decoder_bias_before_encode = subtract_decoder_bias_before_encode

        # Encoder projects d_input -> d_dict.
        self.W_enc = nn.Parameter(torch.empty(d_input, d_dict))
        # Decoder bias absorbs the input mean.
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        if untie_weights:
            self.W_dec = nn.Parameter(torch.empty(d_dict, d_input))
        else:
            # Tied case: we don't allocate W_dec; forward() uses W_enc.T.
            self.register_parameter("W_dec", None)

        self._init_weights()

        # Running count of how often each dictionary atom fires — used by
        # the training loop to identify and revive "dead" neurons.
        self.register_buffer("firing_count", torch.zeros(d_dict, dtype=torch.long))
        self.register_buffer("steps_since_reset", torch.zeros(1, dtype=torch.long))

    def _init_weights(self) -> None:
        # Kaiming uniform on the encoder — standard for MLPs.
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        if self.untie_weights:
            # Untied decoder starts as encoder.T (warm start) then diverges.
            with torch.no_grad():
                self.W_dec.copy_(self.W_enc.T)
        # Normalize decoder-row norms to 1 — a standard SAE trick that
        # prevents the model from cheating by scaling feature magnitudes.
        self._normalize_decoder_rows()

    def _decoder_weight(self) -> torch.Tensor:
        return self.W_dec if self.untie_weights else self.W_enc.T

    @torch.no_grad()
    def _normalize_decoder_rows(self) -> None:
        W = self._decoder_weight()
        # Each row of W is a dictionary atom (in input space); normalize.
        norms = torch.linalg.norm(W, dim=1, keepdim=True).clamp(min=1e-8)
        W.div_(norms)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode activations -> (topk_values, topk_indices).

        Returns the sparse activation as a pair of tensors rather than a
        materialized dense (batch, d_dict) matrix — cheaper for very
        wide dictionaries. Callers who want the dense form can build it
        via `scatter_` (see `encode_dense`).
        """
        if self.subtract_decoder_bias_before_encode:
            x = x - self.b_dec
        pre = x @ self.W_enc                            # (batch, d_dict)
        pre = torch.relu(pre)                            # non-negative activations
        values, indices = torch.topk(pre, k=self.k, dim=-1)
        return values, indices

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Dense (batch, d_dict) activation with only top-k non-zero entries.

        Convenience wrapper — most callers should prefer `encode()` for
        memory efficiency, but tests / feature-interpretation code often
        wants the dense form.
        """
        values, indices = self.encode(x)
        z = torch.zeros(
            x.shape[0], self.d_dict, device=x.device, dtype=x.dtype
        )
        z.scatter_(dim=-1, index=indices, src=values)
        return z

    def decode(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from sparse (values, indices) encoding."""
        W_dec = self._decoder_weight()
        # For each of the k selected atoms, look up its decoder row and
        # scale it by the activation. gather along dim=0 of W_dec.
        # W_dec: (d_dict, d_input) -> selected: (batch, k, d_input)
        selected = W_dec[indices]
        # Weight each selected atom by its activation, sum over k.
        contributions = selected * values.unsqueeze(-1)
        recon = contributions.sum(dim=1) + self.b_dec
        return recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode + decode. Returns (reconstruction, values, indices).

        The values/indices are returned so the training loop can update
        firing counts and compute the auxiliary loss without re-encoding.
        """
        values, indices = self.encode(x)
        recon = self.decode(values, indices)
        self._update_firing_stats(indices)
        return recon, values, indices

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Per-batch mean-squared reconstruction error."""
        return ((x - x_hat) ** 2).mean()

    def aux_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        n_dead_to_revive: int = 128,
    ) -> torch.Tensor:
        """Dead-neuron revival loss (Gao et al. section 3.4).

        Picks the `n_dead_to_revive` least-frequently-firing dictionary
        atoms and asks them to predict the reconstruction residual. This
        gives dead atoms a gradient signal, dragging them back into use.

        Returns 0 if fewer than n_dead_to_revive atoms qualify as "dead"
        (defined as firing_count == 0 since last reset).
        """
        residual = x - x_hat
        dead_mask = self.firing_count == 0
        n_dead = int(dead_mask.sum().item())
        if n_dead < n_dead_to_revive:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Compute activation of dead atoms on the current batch.
        if self.subtract_decoder_bias_before_encode:
            centered = x - self.b_dec
        else:
            centered = x
        dead_atom_ids = torch.nonzero(dead_mask, as_tuple=False).squeeze(-1)
        dead_pre = torch.relu(centered @ self.W_enc[:, dead_atom_ids])
        # Take top-`n_dead_to_revive` of the dead atoms by activation magnitude.
        vals, idxs = torch.topk(dead_pre, k=n_dead_to_revive, dim=-1)
        # Reconstruct residual using those atoms only.
        W_dec = self._decoder_weight()
        selected_atom_ids = dead_atom_ids[idxs]
        selected = W_dec[selected_atom_ids]
        contributions = selected * vals.unsqueeze(-1)
        residual_hat = contributions.sum(dim=1)
        return ((residual - residual_hat) ** 2).mean()

    @torch.no_grad()
    def _update_firing_stats(self, indices: torch.Tensor) -> None:
        """Increment firing_count for each active atom on this batch."""
        flat = indices.reshape(-1)
        self.firing_count.index_add_(0, flat, torch.ones_like(flat, dtype=torch.long))
        self.steps_since_reset += 1

    @torch.no_grad()
    def reset_firing_stats(self) -> None:
        """Zero out firing counts — called at the start of each epoch."""
        self.firing_count.zero_()
        self.steps_since_reset.zero_()

    def dead_atom_fraction(self) -> float:
        """Fraction of dictionary atoms that have not fired since the last reset."""
        return float((self.firing_count == 0).sum().item()) / self.d_dict
