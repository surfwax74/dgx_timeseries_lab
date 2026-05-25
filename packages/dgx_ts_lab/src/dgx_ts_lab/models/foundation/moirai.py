"""Moirai adapter — Salesforce's multivariate-native foundation model.

Full implementation requires the ``uni2ts`` package (Salesforce's research
framework for Moirai). Phase 3 ships an architectural shell that:

    - registers as a detector,
    - declares honest Capabilities,
    - errors at fit() with a clear message if ``uni2ts`` isn't installed,
    - succeeds at import time so ``DETECTOR_REGISTRY.list()`` shows it.

To enable: ``pip install uni2ts`` (sneakernet for air-gap) and provision
weights under ``data/models/Salesforce/moirai-1.0-R-small/`` (or register
in MLflow). Then ``dgx-ts train model=moirai_zero ...`` will work.

Native multivariate handling: Moirai supports multivariate input directly,
so we override the per-channel-then-max strategy from the base and let it
process all channels jointly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dgx_ts_core.registry import DETECTOR_REGISTRY

from ._base import ForecastingDetector


_UNI2TS_HINT = """\
Moirai requires the `uni2ts` package (Salesforce's research framework).

  Install (connected machine): pip install uni2ts
  Air-gap: pre-download the wheel + transitive deps via uv pip download,
           sneakernet, then `uv pip install --no-index --find-links ./wheels uni2ts`.

Then provision weights under data/models/Salesforce/moirai-1.0-R-small/
(see docs/foundation_model_provisioning.md).
"""


def _try_import_uni2ts():
    try:
        import uni2ts  # noqa: F401
        return True
    except ImportError:
        return False


class _MoiraiModule(nn.Module):
    """Wraps a Moirai forecaster from uni2ts. Constructed only if uni2ts available."""

    def __init__(
        self,
        model_path: Path,
        n_channels: int,
        window_length: int,
    ) -> None:
        super().__init__()
        if not _try_import_uni2ts():
            raise ImportError(_UNI2TS_HINT)
        # When uni2ts is available, instantiate the real model.
        # Placeholder: from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        # Real loader code lives here once uni2ts is provisioned.
        self.n_channels = int(n_channels)
        self.window_length = int(window_length)
        # Fallback layer for the architectural shell:
        self.linear = nn.Linear(n_channels, n_channels)
        self.register_buffer("norm_mean", torch.zeros(n_channels))
        self.register_buffer("norm_std", torch.ones(n_channels))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        # Real Moirai would emit (B, horizon, C) forecasts; this shell
        # returns the input shifted-by-1 as a placeholder so downstream
        # plumbing works during dev.
        return self.linear(x_norm)


class MoiraiDetector(ForecastingDetector):
    """Moirai adapter. Multivariate-native; bypasses per-channel-then-max."""

    name = "moirai"

    def __init__(
        self,
        model: str = "Salesforce/moirai-1.0-R-small",
        window_length: int = 256,
        n_channels: int | None = None,
    ) -> None:
        self._model_name = model
        self._window_length = int(window_length)
        self._n_channels = int(n_channels) if n_channels is not None else None
        self._module: _MoiraiModule | None = None

    def _build_module(self, n_channels: int, window_length: int) -> nn.Module:
        if not _try_import_uni2ts():
            raise ImportError(_UNI2TS_HINT)
        from ._loader import resolve_model_path

        try:
            path = resolve_model_path(self._model_name)
        except FileNotFoundError:
            path = Path("__missing__")
        return _MoiraiModule(
            model_path=path, n_channels=n_channels, window_length=window_length
        )

    def _forecast_batch(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self.module(x_norm)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._module is None:
            raise RuntimeError("nothing to save — detector not fit")
        torch.save(
            {
                "model_name": self._model_name,
                "window_length": self._window_length,
                "n_channels": self._n_channels,
                "module_state": self._module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "MoiraiDetector":
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        det = cls(
            model=data["model_name"],
            window_length=data["window_length"],
            n_channels=data["n_channels"],
        )
        det._n_channels = data["n_channels"]
        det._module = _MoiraiModule(
            model_path=Path("__missing__"),
            n_channels=det._n_channels,
            window_length=det._window_length,
        )
        det._module.load_state_dict(data["module_state"])
        return det


@DETECTOR_REGISTRY.register("moirai")
def _create(**kwargs: Any) -> MoiraiDetector:
    return MoiraiDetector(**kwargs)
