# dgx_ts_lab.datasets.synthetic.layered

Composable L1–L6 generator. Each layer is a `Component` with `apply(state, rng)`; the orchestrator chains them in declared order.

## Files

| File | Layer | Components |
|---|---|---|
| `component.py` | base | `GenState`, `Component` |
| `physics.py` | L1 | `OrbitalSinusoid`, `SolarIllumination`, `ThermalDutyCycle`, `BatterySoC` |
| `modes.py` | L2 | `ModeMachine` + `MODE_VOCAB` constants |
| `coupling.py` | L3 | `LinearCoupling`, `InverseCoupling` |
| `noise.py` | L4 | `GaussianNoise`, `PinkNoise`, `StudentTNoise`, `QuantizationNoise`, `CorrelatedGaussianNoise`, `PoissonBurstNoise`, `MultiplicativeGainNoise` |
| `drift.py` | L5 | `LinearDrift`, `ExponentialAging`, `SeasonalModulation`, `RegimeChange` |
| `faults.py` | L6 | `PointFault`, `DropoutFault`, `StuckAtFault`, `DriftFault`, `OscillationFault`, `CorrelationBreakFault`, `ModeConfusionFault` |
| `orchestrator.py` | — | `LayeredSyntheticDataset` (implements `TelemetryDataset`); registered as `layered_synth`. |

## Order matters

The orchestrator applies components **in the order you supply them**. Recommended ordering:

1. **L2 modes first** so downstream physics can branch on them.
2. **L1 physics** (writes deterministic baseline signals).
3. **L3 coupling** (cross-channel correlations).
4. **L4 noise** (each channel gets its own noise budget).
5. **L5 drift / non-stationarity**.
6. **L6 faults last** so labels reflect the full additive picture.

## Determinism

Single `rng` seeded from `LayeredSyntheticDataset.seed` is threaded through every component. Same config + same seed → byte-identical output.

## Adding a new component

1. Subclass `Component` in the appropriate layer file (or create a new one for a new layer).
2. Override `apply(self, state, rng) -> None`. Mutate `state.data`, `state.mode`, `state.labels` in place.
3. For faults: also append a structured entry to `state.fault_log`.
4. Add a unit test in [`packages/dgx_ts_lab/tests/test_layered_components.py`](../../../../../tests/test_layered_components.py).

## See also

- Parent: [`../README.md`](../README.md) (synthetic overview, both trivial + layered)
- Preset config: [`configs/dataset/presets/leo_eps_24h.yaml`](../../../../../../../configs/dataset/presets/leo_eps_24h.yaml)
