"""SequenceTransformer — BERT-style MLM detector for command sequences.

Phase 8 cybersecurity detector. Token-embed + positional-embed + standard
transformer encoder + MLM head. Anomaly score per token = prediction
perplexity, averaged over K random mask samples for stability.

Reads token IDs from `batch["x"]` (shape `(B, T, 1)` float32, cast to long
internally). Designed to work with `CommandSequenceDataset` out of the box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgx_ts_core.data import TelemetryDataset, TelemetryWindow
from dgx_ts_core.models import (
    AnomalyScore,
    Capabilities,
    FitMode,
    FitResult,
    OutputKind,
)
from dgx_ts_core.registry import DETECTOR_REGISTRY

from ...datasets.cyber._tokenizer import MASK_TOKEN, N_SPECIAL, PAD_TOKEN


class SequenceTransformerModule(nn.Module):
    """nn.Module for BERT-style MLM over command tokens."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 1024,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self.d_model = int(d_model)

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=PAD_TOKEN)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_seq_len, self.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.mlm_head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T) long. Returns logits (B, T, vocab).
        T = token_ids.shape[1]
        if T > self.max_seq_len:
            token_ids = token_ids[:, : self.max_seq_len]
            T = self.max_seq_len
        emb = self.token_emb(token_ids) + self.pos_emb[:, :T, :]
        encoded = self.encoder(emb)
        return self.mlm_head(encoded)


def _mlm_mask(
    token_ids: torch.Tensor, mask_prob: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Random Bernoulli mask, excluding PAD + special tokens."""
    rand = torch.rand(token_ids.shape, device=token_ids.device, generator=generator)
    mask = rand < mask_prob
    # Don't mask special tokens (PAD/CMD/UNK/MASK) — they're structural.
    mask = mask & (token_ids >= N_SPECIAL)
    return mask


class SequenceTransformerDetector:
    """AnomalyDetector wrapping SequenceTransformerModule."""

    def __init__(
        self,
        vocab_size: int = 32,
        max_seq_len: int = 512,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        mask_prob: float = 0.15,
        score_n_samples: int = 5,
    ) -> None:
        self._cfg: dict[str, Any] = dict(
            vocab_size=int(vocab_size),
            max_seq_len=int(max_seq_len),
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            d_ff=int(d_ff),
            dropout=float(dropout),
        )
        self._mask_prob = float(mask_prob)
        self._score_n_samples = int(score_n_samples)
        self.module: SequenceTransformerModule | None = None

    @property
    def name(self) -> str:
        return "sequence_transformer"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            requires_pretraining=True,
            supports_streaming=False,
            supports_multivariate=False,    # single-channel token stream
            native_context_len=self._cfg["max_seq_len"],
            output_kind=OutputKind.PER_STEP,
            supports_peft=False,
            supports_export_onnx=False,     # MLM head + random masking complicates trace
        )

    # ── neural-detector contract ────────────────────────────────────────

    @staticmethod
    def _tokens_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # batch["x"] is (B, T, 1) float32 — cast back to long token IDs.
        x = batch["x"]
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x.long()

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        token_ids = self._tokens_from_batch(batch)
        mask = _mlm_mask(token_ids, self._mask_prob)
        if not mask.any():
            # No maskable tokens this batch — return zero with grad
            return self.module.token_emb.weight.sum() * 0.0
        masked = torch.where(mask, torch.full_like(token_ids, MASK_TOKEN), token_ids)
        logits = self.module(masked)
        return F.cross_entropy(
            logits[mask],
            token_ids[mask],
        )

    def compute_score_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        token_ids = self._tokens_from_batch(batch)
        B, T = token_ids.shape
        self.module.eval()
        accum = torch.zeros(B, T, device=token_ids.device, dtype=torch.float32)
        counts = torch.zeros(B, T, device=token_ids.device, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(self._score_n_samples):
                mask = _mlm_mask(token_ids, self._mask_prob)
                if not mask.any():
                    continue
                masked = torch.where(mask, torch.full_like(token_ids, MASK_TOKEN), token_ids)
                logits = self.module(masked)
                log_probs = F.log_softmax(logits, dim=-1)
                # Per-position negative log-likelihood of the TRUE token
                nll = -log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
                accum = accum + nll * mask.float()
                counts = counts + mask.float()
        # Avoid divide-by-zero on positions never masked
        scores = torch.where(counts > 0, accum / counts.clamp_min(1.0), torch.zeros_like(accum))
        return scores

    # ── AnomalyDetector Protocol ────────────────────────────────────────

    def fit(
        self,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: dict[str, Any],
    ) -> FitResult:
        # If the dataset declares a vocab size, override the config.
        vocab = int(getattr(dataset, "vocab_size", self._cfg["vocab_size"]))
        if vocab > self._cfg["vocab_size"]:
            self._cfg["vocab_size"] = vocab
        if self.module is None:
            self.module = SequenceTransformerModule(**self._cfg)
        return FitResult(
            detector_name=self.name,
            mode=mode,
            n_steps=0,
            metadata={
                "n_parameters": sum(p.numel() for p in self.module.parameters()),
                "vocab_size": self._cfg["vocab_size"],
                "mask_prob": self._mask_prob,
            },
        )

    def score(self, window: TelemetryWindow) -> AnomalyScore:
        if self.module is None:
            raise RuntimeError("must fit before scoring")
        device = next(self.module.parameters()).device
        x = torch.from_numpy(window.tensor).float().unsqueeze(0).to(device)
        scores = (
            self.compute_score_batch({"x": x}).squeeze(0).cpu().numpy().astype(np.float32)
        )
        return AnomalyScore(scores=scores)

    def embed(self, window):
        raise NotImplementedError

    def reconstruct(self, window):
        raise NotImplementedError("sequence_transformer scores via perplexity, not reconstruction")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.module is None:
            raise RuntimeError("nothing to save — detector not fit")
        torch.save(
            {
                "config": self._cfg,
                "mask_prob": self._mask_prob,
                "score_n_samples": self._score_n_samples,
                "module_state": self.module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SequenceTransformerDetector":
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(
            mask_prob=data["mask_prob"],
            score_n_samples=data["score_n_samples"],
            **data["config"],
        )
        det.module = SequenceTransformerModule(**det._cfg)
        det.module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("sequence_transformer")
def _create(**kwargs: Any) -> SequenceTransformerDetector:
    return SequenceTransformerDetector(**kwargs)
