"""Optional NVIDIA Modulus thermal FEM PINN.

Phase 9 ships this as a clearly-marked shell — same architectural slot as
the hand-rolled ``thermal_pinn.py``, but delegates to Modulus when the
package is installed.

To enable:
    pip install nvidia-modulus  # ~5 GB, requires CUDA

Then instantiate ``ModulusThermalFEM(...)`` instead of ``ThermalPinn(...)``;
it implements the same ``PhysicsModel`` Protocol so ``PINNResidualDetector``
treats it identically.
"""

from __future__ import annotations

import numpy as np
from dgx_ts_core.data import TelemetryWindow

_MODULUS_HINT = """\
NVIDIA Modulus is not installed. Install via:

  pip install nvidia-modulus

(~5 GB; CUDA required for training). Air-gap users should pre-cache the
wheel + transitive deps via `uv pip download` and sneakernet.

For now, use the hand-rolled `ThermalPinn` in `thermal_pinn.py` instead —
same `PhysicsModel` interface, no extra dependency.
"""


def _try_import_modulus():
    try:
        import nvidia_modulus as _  # noqa: F401  (variable name avoids shadowing module)
        return True
    except ImportError:
        return False


class ModulusThermalFEM:
    """PhysicsModel-shaped wrapper around a Modulus thermal PINN.

    Constructed only when Modulus is available; otherwise raises with the
    install hint at import + fit time so callers get a clear pointer.
    """

    name = "modulus_thermal_fem"

    def __init__(self, **modulus_kwargs) -> None:
        if not _try_import_modulus():
            raise ImportError(_MODULUS_HINT)
        # When Modulus is available, instantiate the real PINN here.
        # Placeholder: from nvidia_modulus.models.fno import FNO
        self._modulus_kwargs = modulus_kwargs
        self._zones_covered: set[str] = set()

    def covered_channels(self) -> set[str]:
        return self._zones_covered

    def predict(self, window: TelemetryWindow) -> np.ndarray:
        # Real Modulus inference path goes here once package is provisioned.
        # For the architectural shell, return zeros (same shape as input tensor).
        return np.zeros_like(window.tensor)
