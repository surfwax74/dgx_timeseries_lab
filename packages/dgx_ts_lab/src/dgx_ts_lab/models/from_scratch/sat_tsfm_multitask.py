"""SatTSFM multi-task wrapper — shared encoder + composable heads.

Phase 6 detector. Reuses ``SatTSFMModule`` as the encoder backbone, attaches
one or more ``TaskHead`` instances on top of per-step pooled embeddings,
and sums weighted losses for joint training. AD score continues to come
from the base detector's reconstruction error (so this wrapper subsumes
the AD task without re-implementing it).

Locked Phase 6 decision: shared encoder + joint multi-task loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dgx_ts_core.data import TelemetryDataset, TelemetryWindow
from dgx_ts_core.models import (
    AnomalyScore,
    Capabilities,
    FitMode,
    FitResult,
    OutputKind,
)
from dgx_ts_core.registry import DETECTOR_REGISTRY, HEAD_REGISTRY

from ..heads._base import TaskHead
from .sat_tsfm import SatTSFMDetector, SatTSFMModule


class _SatTSFMMultiTaskModule(nn.Module):
    """Wraps SatTSFMModule + a ModuleDict of heads. The heads see the
    per-step pooled embeddings produced by ``base.encode_pooled_steps()``.
    """

    def __init__(self, base: SatTSFMModule, heads: dict[str, TaskHead]) -> None:
        super().__init__()
        self.base = base
        self.heads = nn.ModuleDict(heads)
        # Surface norm buffers at the module level so Fabric / serving
        # code that introspects `module.norm_mean` keeps working.

    @property
    def norm_mean(self):
        return self.base.norm_mean

    @property
    def norm_std(self):
        return self.base.norm_std


class SatTSFMMultiTaskDetector:
    """Multi-task detector wrapping a Sat-TSFM backbone + attached heads."""

    def __init__(
        self,
        # SatTSFM backbone config (forwarded to SatTSFMDetector)
        window_length: int = 256,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_channels: int = 256,
        n_channels: int | None = None,
        # Heads config: list of {key: <head_registry_key>, params: {...}}
        heads: list[dict[str, Any]] | None = None,
        # Loss weights
        ad_loss_weight: float = 1.0,
    ) -> None:
        self._base_detector = SatTSFMDetector(
            window_length=window_length,
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_channels=max_channels,
            n_channels=n_channels,
        )
        self._heads_cfg: list[dict[str, Any]] = list(heads or [])
        self._ad_loss_weight = float(ad_loss_weight)
        self._n_channels = int(n_channels) if n_channels is not None else None
        self._d_model = int(d_model)
        self.module: _SatTSFMMultiTaskModule | None = None
        self._head_objs: dict[str, TaskHead] = {}

    @property
    def name(self) -> str:
        return "sat_tsfm_multitask"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            requires_pretraining=True,
            supports_streaming=False,
            supports_multivariate=True,
            native_context_len=self._base_detector._cfg["window_length"],
            output_kind=OutputKind.PER_STEP,
            supports_peft=True,
            supports_export_onnx=False,    # multiple outputs complicate trace; Phase 5 follow-up
            supports_multi_task=True,
        )

    # ── shared helpers ───────────────────────────────────────────────────

    def _ad_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction MSE (same shape as base SatTSFM's compute_loss)."""
        assert self.module is not None
        base = self.module.base
        recon = base(x)
        C = x.shape[-1]
        ids = torch.arange(C, device=x.device, dtype=torch.long)
        m = base.norm_mean[ids].view(1, 1, C)
        s = base.norm_std[ids].view(1, 1, C)
        x_norm = (x - m) / s
        return ((recon - x_norm) ** 2).mean()

    def _encoded_per_step(self, x: torch.Tensor) -> torch.Tensor:
        assert self.module is not None
        return self.module.base.encode_pooled_steps(x)

    def _pad_or_crop_aux(
        self, aux: dict[str, torch.Tensor] | None, T: int
    ) -> dict[str, torch.Tensor] | None:
        """Align aux_label step counts to T (the encoded step count)."""
        if aux is None:
            return None
        out: dict[str, torch.Tensor] = {}
        for k, v in aux.items():
            if v.dim() < 1:
                out[k] = v
            elif v.shape[-1] == T:
                out[k] = v
            elif v.shape[-1] > T:
                # Crop trailing steps (encoder truncates to n_patches*patch_len)
                out[k] = v[..., :T]
            else:
                # Pad with sentinel — head logic should handle (-1 for ints)
                pad = T - v.shape[-1]
                if v.dtype.is_floating_point:
                    fill = float("nan")
                else:
                    fill = -1
                pad_t = torch.full((*v.shape[:-1], pad), fill, dtype=v.dtype, device=v.device)
                out[k] = torch.cat([v, pad_t], dim=-1)
        return out

    # ── neural-detector contract ─────────────────────────────────────────

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        x = batch["x"]
        total = self._ad_loss_weight * self._ad_loss(x)
        if self._head_objs and batch.get("aux_labels") is not None:
            encoded = self._encoded_per_step(x)             # (B, T_trunc, D)
            T_trunc = encoded.shape[1]
            head_batch = {
                **batch,
                "aux_labels": self._pad_or_crop_aux(batch["aux_labels"], T_trunc),
            }
            for name, head in self._head_objs.items():
                try:
                    head_loss = head.compute_loss(encoded, head_batch)
                    total = total + head.loss_weight * head_loss
                except KeyError:
                    # Head asked for a label that this batch doesn't have — skip.
                    continue
        return total

    def compute_score_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # AD scoring is identical to the base detector — reuse it.
        return self._base_detector.compute_score_batch(batch)

    # ── per-task auxiliary methods (called by tests / eval CLI) ─────────

    def per_task_outputs(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return raw outputs per head for a batch (for eval / inspection)."""
        if self.module is None:
            raise RuntimeError("module not built — call fit() first")
        self.module.eval()
        with torch.no_grad():
            encoded = self._encoded_per_step(batch["x"])
            return {name: head(encoded) for name, head in self._head_objs.items()}

    def per_task_metrics(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        if self.module is None or not self._head_objs:
            return {}
        encoded = self._encoded_per_step(batch["x"])
        T_trunc = encoded.shape[1]
        head_batch = {
            **batch,
            "aux_labels": self._pad_or_crop_aux(batch.get("aux_labels"), T_trunc),
        }
        out: dict[str, float] = {}
        for name, head in self._head_objs.items():
            try:
                out.update(head.compute_metrics(encoded, head_batch))
            except KeyError:
                continue
        return out

    # ── AnomalyDetector Protocol ────────────────────────────────────────

    def fit(
        self,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: dict[str, Any],
    ) -> FitResult:
        # Initialize base + heads on first call.
        base_result = self._base_detector.fit(dataset, mode, config)
        self._n_channels = self._base_detector._n_channels

        if self.module is None:
            heads: dict[str, TaskHead] = {}
            for head_spec in self._heads_cfg:
                spec = dict(head_spec)
                key = spec.pop("key")
                spec.setdefault("d_model", self._d_model)
                head = HEAD_REGISTRY.create(key, **spec)
                heads[key] = head
            self._head_objs = heads
            assert self._base_detector.module is not None
            self.module = _SatTSFMMultiTaskModule(self._base_detector.module, heads)

        n_params_total = sum(p.numel() for p in self.module.parameters())
        n_params_base = sum(p.numel() for p in self.module.base.parameters())
        n_params_heads = n_params_total - n_params_base
        return FitResult(
            detector_name=self.name,
            mode=mode,
            n_steps=base_result.n_steps,
            metadata={
                "n_parameters_total": n_params_total,
                "n_parameters_base": n_params_base,
                "n_parameters_heads": n_params_heads,
                "head_keys": list(self._head_objs.keys()),
                "ad_loss_weight": self._ad_loss_weight,
            },
        )

    def score(self, window: TelemetryWindow) -> AnomalyScore:
        return self._base_detector.score(window)

    def embed(self, window):
        raise NotImplementedError

    def reconstruct(self, window):
        return self._base_detector.reconstruct(window)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.module is None:
            raise RuntimeError("nothing to save — detector not fit")
        torch.save(
            {
                "base_config": self._base_detector._cfg,
                "n_channels": self._n_channels,
                "heads_cfg": self._heads_cfg,
                "ad_loss_weight": self._ad_loss_weight,
                "module_state": self.module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SatTSFMMultiTaskDetector":
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(
            n_channels=data["n_channels"],
            heads=data["heads_cfg"],
            ad_loss_weight=data["ad_loss_weight"],
            **data["base_config"],
        )
        det._n_channels = data["n_channels"]
        det._base_detector._n_channels = data["n_channels"]
        det._base_detector.module = SatTSFMModule(**data["base_config"])
        heads: dict[str, TaskHead] = {}
        for head_spec in det._heads_cfg:
            spec = dict(head_spec)
            key = spec.pop("key")
            spec.setdefault("d_model", det._d_model)
            heads[key] = HEAD_REGISTRY.create(key, **spec)
        det._head_objs = heads
        det.module = _SatTSFMMultiTaskModule(det._base_detector.module, heads)
        det.module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("sat_tsfm_multitask")
def _create(**kwargs: Any) -> SatTSFMMultiTaskDetector:
    return SatTSFMMultiTaskDetector(**kwargs)
