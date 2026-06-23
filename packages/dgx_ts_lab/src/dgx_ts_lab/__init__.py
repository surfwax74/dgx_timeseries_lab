"""dgx_ts_lab — implementations of the dgx_ts_core contracts.

Importing this package triggers self-registration of all bundled datasets,
detectors, and trainers with their respective registries.
"""

import warnings as _warnings

# ── Quiet a harmless probe-warning from torch on CUDA-only boxes ───────
# torch.xpu (Intel GPU backend) warns "XPU device count is zero!" whenever
# accelerator auto-detection runs on a box without Intel GPUs — i.e. every
# DGX H200 node. The warning is informational; CUDA detection still works.
# We filter it here (narrow, by message text) rather than globally so any
# other torch UserWarnings still surface.
_warnings.filterwarnings(
    "ignore",
    message=r".*XPU device count is zero.*",
    category=UserWarning,
)

__version__ = "0.1.0"

# Side-effect imports — populate the registries
from . import datasets, llm, models, training  # noqa: F401, E402

__all__ = ["datasets", "llm", "models", "training"]
