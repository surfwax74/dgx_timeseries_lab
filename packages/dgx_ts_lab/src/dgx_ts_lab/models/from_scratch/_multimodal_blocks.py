"""Cross-modal attention building blocks for Phase 10 SatMultiModal.

Three pieces:
    ModalityTypeEmbedding   per-modality learnable bias added to token embeddings
    PerModalitySelfAttn     small stack of standard transformer encoder layers
    SharedCrossModalBlock   standard transformer encoder layer used in the
                            cross-modal stack (where modalities mix)

Architecture pattern (locked Phase 10): per-modality self-attn → concat with
modality embedding → 2-3 shared cross-modal blocks. Matches Perceiver/
Flamingo-style designs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityTypeEmbedding(nn.Module):
    """Add a learnable per-modality embedding to token embeddings.

    Modality IDs: 0 = telemetry, 1 = command, 2 = log.
    """

    TELEMETRY = 0
    COMMAND = 1
    LOG = 2

    def __init__(self, n_modalities: int = 3, d_model: int = 128) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_modalities, d_model)
        nn.init.trunc_normal_(self.emb.weight, std=0.02)

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D); modality_ids: (T,) long → returns (B, T, D) with per-token bias added."""
        return x + self.emb(modality_ids).unsqueeze(0)


class PerModalitySelfAttn(nn.Module):
    """Small transformer-encoder stack applied within one modality."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SharedCrossModalStack(nn.Module):
    """Shared transformer stack run over the CONCATENATED modality tokens."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
