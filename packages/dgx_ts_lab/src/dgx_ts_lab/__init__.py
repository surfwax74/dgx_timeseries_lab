"""dgx_ts_lab — implementations of the dgx_ts_core contracts.

Importing this package triggers self-registration of all bundled datasets,
detectors, and trainers with their respective registries.
"""

__version__ = "0.1.0"

# Side-effect imports — populate the registries
from . import datasets, llm, models, training  # noqa: F401, E402

__all__ = ["datasets", "llm", "models", "training"]
