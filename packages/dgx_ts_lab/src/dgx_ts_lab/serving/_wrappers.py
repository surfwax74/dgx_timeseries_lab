"""ONNX wrapper factories for each supported detector.

Each factory returns a dict of nn.Module wrappers, one per artifact
filename. The base exporter traces them through torch.onnx.export.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .onnx_export import register_onnx_wrapper

# ── PatchTST+MAE ────────────────────────────────────────────────────────


class _PatchTSTRawWrapper(nn.Module):
    """ONNX-traceable raw-score forward for PatchTSTMAEDetector."""

    def __init__(self, module) -> None:
        super().__init__()
        self.m = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → return (B, T) raw per-step scores
        recon_norm, _ = self.m(x, apply_mask=False)
        x_norm = self.m.normalize(x)
        return ((recon_norm - x_norm) ** 2).mean(dim=-1)


class _PatchTSTBakedWrapper(nn.Module):
    """Threshold-baked is_anomaly forward."""

    def __init__(self, module, threshold: float) -> None:
        super().__init__()
        self.m = module
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recon_norm, _ = self.m(x, apply_mask=False)
        x_norm = self.m.normalize(x)
        scores = ((recon_norm - x_norm) ** 2).mean(dim=-1)
        return (scores > self.threshold).to(torch.bool)


def _patchtst_factory(detector, threshold):
    from dgx_ts_lab.models.from_scratch.patchtst_mae import PatchTSTMAEDetector

    assert isinstance(detector, PatchTSTMAEDetector)
    if detector.module is None:
        raise RuntimeError("detector not fit — call fit() before export")
    out: dict[str, nn.Module] = {"model": _PatchTSTRawWrapper(detector.module)}
    if threshold is not None:
        out["model_with_threshold"] = _PatchTSTBakedWrapper(detector.module, threshold)
    return out


# Lazy registration — detector class import is deferred to factory call time
# to avoid circular import at module load.
def _register():
    from dgx_ts_lab.models.from_scratch.patchtst_mae import PatchTSTMAEDetector
    from dgx_ts_lab.models.from_scratch.sat_tsfm import SatTSFMDetector

    register_onnx_wrapper(PatchTSTMAEDetector)(_patchtst_factory)
    register_onnx_wrapper(SatTSFMDetector)(_sat_tsfm_factory)


# ── Sat-TSFM ────────────────────────────────────────────────────────────


class _SatTSFMRawWrapper(nn.Module):
    def __init__(self, module) -> None:
        super().__init__()
        self.m = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate the per-channel-then-max scoring inside the module
        # using only ops ONNX can trace.
        recon = self.m(x)  # (B, T, C) in normalized space
        # Match _normalize done in compute_score_batch
        # (re-normalize x with the buffers)
        C = x.shape[-1]
        ids = torch.arange(C, device=x.device, dtype=torch.long)
        mean = self.m.norm_mean[ids].view(1, 1, C)
        std = self.m.norm_std[ids].view(1, 1, C)
        x_norm = (x - mean) / std
        return ((recon - x_norm) ** 2).max(dim=-1).values   # (B, T)


class _SatTSFMBakedWrapper(nn.Module):
    def __init__(self, module, threshold: float) -> None:
        super().__init__()
        self.m = module
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.m(x)
        C = x.shape[-1]
        ids = torch.arange(C, device=x.device, dtype=torch.long)
        mean = self.m.norm_mean[ids].view(1, 1, C)
        std = self.m.norm_std[ids].view(1, 1, C)
        x_norm = (x - mean) / std
        scores = ((recon - x_norm) ** 2).max(dim=-1).values
        return (scores > self.threshold).to(torch.bool)


def _sat_tsfm_factory(detector, threshold):
    from dgx_ts_lab.models.from_scratch.sat_tsfm import SatTSFMDetector

    assert isinstance(detector, SatTSFMDetector)
    if detector.module is None:
        raise RuntimeError("detector not fit — call fit() before export")
    out: dict[str, nn.Module] = {"model": _SatTSFMRawWrapper(detector.module)}
    if threshold is not None:
        out["model_with_threshold"] = _SatTSFMBakedWrapper(detector.module, threshold)
    return out


_register()
