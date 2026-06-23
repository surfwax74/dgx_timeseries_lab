"""Physics-informed (PINN) residual wrappers.

Pre-process mode (locked Phase 4 decision): the physics model produces an
analytical prediction for each channel; the wrapper subtracts that prediction
from the input before passing the residual to an inner neural detector.

Components:

    pinn_base.py        PINNResidualDetector wrapper + PhysicsModel Protocol
    orbital.py          OrbitalResidual — sun-angle / eclipse predictions
    thermal.py          ThermalResidual — first-order thermal model
    battery.py          BatteryResidual — coulomb counting + Nernst-style voltage

Registry keys:
    pinn_residual               the generic wrapper (takes inner + physics from config)
    orbital_residual_<inner>    pre-built combos for common pairings (added later)
"""

from . import (
    adcs,  # noqa: F401  ADCS subpackage with 3 integrators + ADCS PINN
    battery,  # noqa: F401
    orbital,  # noqa: F401
    pinn_base,  # noqa: F401
    thermal,  # noqa: F401
)

# Phase 9: hand-rolled trainable PINN + ADCS dynamics. Imported here so they're
# part of the public API; registered with _PHYSICS_REGISTRY below.
from ._thermal_solver import (  # noqa: F401
    ThermalBus,
    build_thermal_solver,
)
from ._thermal_solver import (  # noqa: F401
    simulate as simulate_thermal,
)
from .battery import BatteryResidual
from .orbital import OrbitalResidual
from .pinn_base import PhysicsModel, PINNResidualDetector
from .thermal import ThermalResidual
from .thermal_pinn import (  # noqa: F401
    ThermalPinn,
    ThermalPinnConfig,
    ThermalPinnPhysicsModel,
)

_PHYSICS_REGISTRY: dict[str, type] = {
    "orbital": OrbitalResidual,
    "thermal": ThermalResidual,
    "battery": BatteryResidual,
}


__all__ = [
    "BatteryResidual",
    "OrbitalResidual",
    "PINNResidualDetector",
    "PhysicsModel",
    "ThermalResidual",
    "battery",
    "orbital",
    "pinn_base",
    "thermal",
]
