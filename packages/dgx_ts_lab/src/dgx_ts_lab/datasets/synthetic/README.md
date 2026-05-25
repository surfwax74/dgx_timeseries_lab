# Synthetic telemetry datasets

Two generators live here:

| Module | Purpose |
|---|---|
| [`trivial.py`](trivial.py) | Phase 0 smoke-test dataset: sine + spike. ~10k samples, runs on CPU in milliseconds. |
| [`layered/`](layered/) | Phase 1 composable generator: L1 physics → L6 faults. The real synthetic-data story. |

## The layered generator

Each layer is a `Component` with `apply(state, rng)`. The orchestrator
chains them in declared order. Order matters — physics first, modes
early so downstream components can branch on them, faults last so labels
reflect the full additive picture.

### Components shipped today

| Layer | File | Components |
|---|---|---|
| L1 Physics | [`physics.py`](layered/physics.py) | `OrbitalSinusoid`, `SolarIllumination`, `ThermalDutyCycle`, `BatterySoC` |
| L2 Modes | [`modes.py`](layered/modes.py) | `ModeMachine` (sun/eclipse + Poisson payload activations) |
| L3 Coupling | [`coupling.py`](layered/coupling.py) | `LinearCoupling` (with lag), `InverseCoupling` |
| L4 Noise | [`noise.py`](layered/noise.py) | `GaussianNoise`, `PinkNoise`, `StudentTNoise`, `QuantizationNoise`, `CorrelatedGaussianNoise`, `PoissonBurstNoise`, `MultiplicativeGainNoise` |
| L5 Non-stationarity | [`drift.py`](layered/drift.py) | `LinearDrift`, `ExponentialAging`, `SeasonalModulation`, `RegimeChange` |
| L6 Faults | [`faults.py`](layered/faults.py) | `PointFault`, `DropoutFault`, `StuckAtFault`, `DriftFault`, `OscillationFault`, `CorrelationBreakFault`, `ModeConfusionFault` |

Each L6 fault writes `state.labels = True` over its window AND appends a
structured entry to `state.fault_log` (type, channel, start, end,
severity, etc.) so evaluation code can break results down per fault
category.

### Determinism

Every component takes the same `rng` from the orchestrator. The whole
dataset is reproducible from `(component_list, n_samples, seed)`.

### Composing your own — Python

```python
from dgx_ts_core.data import Channel, Subsystem, Units
from dgx_ts_lab.datasets.synthetic.layered import (
    LayeredSyntheticDataset, modes, physics, noise, faults,
)

channels = (
    Channel(name="bus_voltage", units=Units.VOLT, subsystem=Subsystem.EPS, sample_rate_hz=1.0),
)
components = [
    modes.ModeMachine(period_s=5400.0, eclipse_fraction=0.35),
    physics.OrbitalSinusoid("bus_voltage", amplitude=0.3, period_s=5400.0, baseline=28.0),
    noise.GaussianNoise("bus_voltage", std=0.02),
    noise.PinkNoise("bus_voltage", std=0.01),
    faults.PointFault("bus_voltage", rate_per_hour=1.0, magnitude=0.5),
]
ds = LayeredSyntheticDataset(channels, components, n_samples=86400, seed=42)
```

### Composing your own — Hydra

The `layered_synth` factory accepts `_target_` dicts for components and
uses `hydra.utils.instantiate` to materialize them. See
[`configs/dataset/presets/leo_eps_24h.yaml`](../../../../../../configs/dataset/presets/leo_eps_24h.yaml)
for the canonical example — six channels covering EPS/TCS/payload with
every layer exercised.

Run it end-to-end with the Phase 1 smoke experiment:

```powershell
uv run dgx-ts train experiment=phase1_layered
```
