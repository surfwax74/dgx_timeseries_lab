"""Higher-fidelity multi-zone thermal solver for Phase 9 PINN ground truth.

Goes beyond the naive single-temperature Euler integrator in
``layered.physics.ThermalDutyCycle`` by modeling:

    - N zones (default 6 = bus faces ±X, ±Y, ±Z), each with mass, heat capacity,
      area, surface optical properties (emissivity / absorptivity)
    - Solar absorption per zone, dependent on sun-vector projection onto zone normal
    - Radiation to deep space (3 K) via Stefan-Boltzmann
    - Conduction between zones via a configurable conductance matrix
    - Internal dissipation (electronics heat) per zone
    - Eclipse handling: zero solar input when inside Earth's shadow

Integration is RK4 over a time window. Pure NumPy — used as the ground-truth
target for the trainable thermal PINN.

References (rough): Gilmore, "Spacecraft Thermal Control Handbook," AIAA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Solar constant (W/m^2) at 1 AU
SOLAR_FLUX_AU = 1361.0
# Stefan-Boltzmann constant (W/m^2/K^4)
SIGMA_BOLTZMANN = 5.670374e-8
# Deep-space temperature (K)
T_DEEP_SPACE = 3.0


@dataclass
class ThermalBus:
    """Multi-zone thermal model of a small satellite bus.

    Default 6-face cube layout. Override fields for custom geometry.
    """

    n_zones: int = 6
    # Outward-pointing unit normals per zone (default: +X, -X, +Y, -Y, +Z, -Z)
    zone_normals: np.ndarray = field(default_factory=lambda: np.eye(6, 3) * np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    ).repeat(6, axis=0)[:6] if False else np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64))
    # Per-zone surface area (m^2)
    zone_areas: np.ndarray = field(
        default_factory=lambda: np.full(6, 0.25, dtype=np.float64)  # 0.5m × 0.5m faces
    )
    # Per-zone thermal mass (kg)
    zone_masses: np.ndarray = field(
        default_factory=lambda: np.full(6, 2.0, dtype=np.float64)
    )
    # Per-zone specific heat capacity (J/kg/K) — aluminum-ish
    zone_cp: np.ndarray = field(
        default_factory=lambda: np.full(6, 900.0, dtype=np.float64)
    )
    # Inter-zone conduction (W/K). conduction_matrix[i,j] = conductance i↔j.
    # Default: each face conducts to its 4 neighbors at 0.5 W/K.
    conduction_matrix: np.ndarray = field(
        default_factory=lambda: _default_cube_conduction_matrix()
    )
    # Per-zone emissivity (radiation out efficiency)
    emissivity: np.ndarray = field(
        default_factory=lambda: np.full(6, 0.85, dtype=np.float64)
    )
    # Per-zone solar absorptivity
    absorptivity: np.ndarray = field(
        default_factory=lambda: np.full(6, 0.6, dtype=np.float64)
    )
    # Per-zone internal dissipation (W) — electronics heat injected here
    internal_diss: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 3.0, 3.0, 2.0, 2.0], dtype=np.float64)
    )

    def __post_init__(self) -> None:
        n = self.n_zones
        # Defensive: enforce shapes
        assert self.zone_normals.shape == (n, 3)
        assert self.zone_areas.shape == (n,)
        assert self.zone_masses.shape == (n,)
        assert self.zone_cp.shape == (n,)
        assert self.conduction_matrix.shape == (n, n)
        assert self.emissivity.shape == (n,)
        assert self.absorptivity.shape == (n,)
        assert self.internal_diss.shape == (n,)


def _default_cube_conduction_matrix() -> np.ndarray:
    """Each face shares an edge with 4 of the other 5 faces; only the opposite
    face has no direct conduction. Default conductance = 0.5 W/K on shared edges."""
    K = 0.5
    M = np.full((6, 6), K, dtype=np.float64)
    # Opposite-face pairs: (0,1), (2,3), (4,5) → no direct conduction
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        M[i, j] = 0.0
        M[j, i] = 0.0
    np.fill_diagonal(M, 0.0)
    return M


# ── Dynamics ───────────────────────────────────────────────────────────


def _dT_dt(
    T: np.ndarray,
    sun_vec: np.ndarray,
    eclipse: float,
    bus: ThermalBus,
) -> np.ndarray:
    """Compute dT/dt for each zone.

    T: (n_zones,) temperatures in K
    sun_vec: (3,) unit vector from spacecraft to Sun in body frame
    eclipse: 1.0 = full eclipse (no solar), 0.0 = full sun
    """
    # Solar absorption per zone
    cos_inc = np.maximum(0.0, bus.zone_normals @ sun_vec)
    solar = bus.zone_areas * bus.absorptivity * SOLAR_FLUX_AU * cos_inc * (1.0 - eclipse)

    # Radiation to deep space
    rad = bus.zone_areas * bus.emissivity * SIGMA_BOLTZMANN * (T**4 - T_DEEP_SPACE**4)

    # Conduction: dQ/dt_i = sum_j K[i,j] * (T_j - T_i)
    diff = T[None, :] - T[:, None]                   # (n, n)
    conduction = (bus.conduction_matrix * diff).sum(axis=1)

    # Total net heat into zone i (W)
    Q = solar + bus.internal_diss + conduction - rad
    return Q / (bus.zone_masses * bus.zone_cp)


def _rk4_step(
    T: np.ndarray,
    sun_vec_t: np.ndarray,
    sun_vec_t_half: np.ndarray,
    sun_vec_t_plus: np.ndarray,
    eclipse_t: float,
    eclipse_t_half: float,
    eclipse_t_plus: float,
    dt: float,
    bus: ThermalBus,
) -> np.ndarray:
    k1 = _dT_dt(T, sun_vec_t, eclipse_t, bus)
    k2 = _dT_dt(T + 0.5 * dt * k1, sun_vec_t_half, eclipse_t_half, bus)
    k3 = _dT_dt(T + 0.5 * dt * k2, sun_vec_t_half, eclipse_t_half, bus)
    k4 = _dT_dt(T + dt * k3, sun_vec_t_plus, eclipse_t_plus, bus)
    return T + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(
    *,
    bus: ThermalBus | None = None,
    duration_s: float = 86400.0,
    dt_s: float = 60.0,
    initial_temp_K: float = 290.0,
    orbital_period_s: float = 5400.0,
    eclipse_fraction: float = 0.35,
    sun_vec_initial: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run the multi-zone thermal solver for ``duration_s``.

    Returns:
        t          (N,) time in seconds
        T          (N, n_zones) temperatures in K
        sun_vecs   (N, 3) sun vector (rotates around z with orbital period)
        eclipse    (N,) eclipse flag (smooth tanh transition)
    """
    bus = bus or ThermalBus()
    n_zones = bus.n_zones

    n_steps = int(duration_s // dt_s) + 1
    t = np.arange(n_steps, dtype=np.float64) * dt_s

    # Sun vector: rotates in the x-y plane at the orbital period
    omega = 2.0 * np.pi / orbital_period_s
    sun_vecs = np.zeros((n_steps, 3), dtype=np.float64)
    sun_vecs[:, 0] = np.cos(omega * t)
    sun_vecs[:, 1] = np.sin(omega * t)

    # Eclipse: smooth indicator centered on the back half of orbit
    pos = (t % orbital_period_s) / orbital_period_s
    d = np.abs(pos - 0.5) - (eclipse_fraction / 2.0)
    eclipse = 0.5 * (1.0 + np.tanh(-d * 50.0))   # 1 in eclipse, 0 in sun

    if sun_vec_initial is not None:
        sun_vecs[0] = sun_vec_initial / np.linalg.norm(sun_vec_initial)

    T = np.empty((n_steps, n_zones), dtype=np.float64)
    T[0] = initial_temp_K

    # Pre-compute half-step sun vectors + eclipse flags for RK4
    for i in range(n_steps - 1):
        t_half = t[i] + 0.5 * dt_s
        sun_half = np.array([np.cos(omega * t_half), np.sin(omega * t_half), 0.0])
        pos_half = (t_half % orbital_period_s) / orbital_period_s
        d_half = abs(pos_half - 0.5) - (eclipse_fraction / 2.0)
        eclipse_half = 0.5 * (1.0 + np.tanh(-d_half * 50.0))
        T[i + 1] = _rk4_step(
            T[i],
            sun_vecs[i], sun_half, sun_vecs[i + 1],
            eclipse[i], eclipse_half, eclipse[i + 1],
            dt_s, bus,
        )

    return {"t": t, "T": T, "sun_vecs": sun_vecs, "eclipse": eclipse}


def build_thermal_solver(**overrides: Any) -> ThermalBus:
    """Convenience: build a ThermalBus with optional field overrides."""
    bus = ThermalBus()
    for k, v in overrides.items():
        if hasattr(bus, k):
            setattr(bus, k, np.asarray(v) if isinstance(v, list) else v)
    return bus
