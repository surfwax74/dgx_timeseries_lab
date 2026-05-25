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
