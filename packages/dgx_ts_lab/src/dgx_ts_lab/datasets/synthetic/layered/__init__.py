"""Layered synthetic telemetry generator.

A composable stack of `Component` objects (L1 physics → L6 faults) drives
generation. Importing this module also registers ``layered_synth`` with
DATASET_REGISTRY.
"""

from . import (  # noqa: F401  side-effect: register layered_synth
    coupling,
    drift,
    faults,
    modes,
    noise,
    orchestrator,
    physics,
)
from .component import Component, GenState
from .orchestrator import LayeredSyntheticDataset

__all__ = [
    "Component",
    "GenState",
    "LayeredSyntheticDataset",
    "coupling",
    "drift",
    "faults",
    "modes",
    "noise",
    "orchestrator",
    "physics",
]
