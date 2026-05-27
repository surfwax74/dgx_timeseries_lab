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

_PHYSICS_REGISTRY: dict[str, type] = {}


def _register_physics() -> None:
    """Populate the physics registry — must run after all submodules import."""
    from .battery import BatteryResidual
    from .orbital import OrbitalResidual
    from .thermal import ThermalResidual

    _PHYSICS_REGISTRY["orbital"] = OrbitalResidual
    _PHYSICS_REGISTRY["thermal"] = ThermalResidual
    _PHYSICS_REGISTRY["battery"] = BatteryResidual


from . import battery, orbital, pinn_base, thermal  # noqa: F401, E402
from .battery import BatteryResidual
from .orbital import OrbitalResidual
from .pinn_base import PhysicsModel, PINNResidualDetector
from .thermal import ThermalResidual

# Phase 9: hand-rolled trainable PINN + ADCS dynamics. Imported here so they're
# part of the public API; registered with _PHYSICS_REGISTRY for Hydra access.
from ._thermal_solver import ThermalBus, build_thermal_solver, simulate as simulate_thermal  # noqa: F401, E402
from .thermal_pinn import ThermalPinn, ThermalPinnConfig, ThermalPinnPhysicsModel  # noqa: F401, E402
from . import adcs  # noqa: F401, E402  ADCS subpackage with 3 integrators + ADCS PINN

_register_physics()


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
