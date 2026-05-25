# dgx_ts_lab.models.physics

Phase 4 physics-informed (PINN) residual wrappers. The pattern: an analytical physics model predicts the channels it knows about; the wrapper subtracts that prediction from the input; a neural detector trains on the residual. Anything physics already explains is removed from the AD job.

## Files

| File | Role | Registry key (physics) |
|---|---|---|
| `pinn_base.py` | `PINNResidualDetector` wrapper + `PhysicsModel` Protocol. Registered as `pinn_residual` (detector). | — |
| `orbital.py` | Sun-angle / eclipse predictions for solar-array channels. | `orbital` |
| `thermal.py` | First-order Euler thermal model for panel / battery / electronics temps. | `thermal` |
| `battery.py` | Coulomb counting + linear V-vs-SoC for battery channels. | `battery` |

## How to use from Hydra

```yaml
# configs/model/orbital_pinn_patchtst.yaml
_target_key: pinn_residual
inner:
  _target_key: patchtst_mae
  d_model: 64
  ...
physics:
  _target_key: orbital
  period_s: 5400.0
```

The PINN wrapper takes:
- `inner`: a dict naming a registered detector + its kwargs
- `physics`: a dict naming a registered physics model + its kwargs

At fit time, the wrapper builds both, constructs a residual dataset (data − physics prediction), and fits the inner detector on it. At score time, it does the same subtraction on each window before scoring.

## Pattern: pre-process mode

This is the **locked Phase 4** design. Alternatives (joint training, post-process re-rank) can be added as separate wrappers without touching this code.

```
input window               physics prediction              residual
(B, T, C)        --        (B, T, C)             =        (B, T, C)
                                                              |
                                                              ↓
                                                       inner detector
                                                              |
                                                              ↓
                                                       anomaly score
```

The inner detector never sees the physics-predictable component. For an ideal physics model, the residual is just noise + actual anomalies — much easier signal to learn.

## When PINN helps

- **High-quality physics + low-quality data**: PINN wins big because residual SNR is much higher than raw SNR.
- **Drift-dominated channels**: PINN removes the drift (which physics knows about) leaving just the anomalies.
- **Cross-mission generalization**: physics is universal; data-driven detectors may overfit a single mission.

## When PINN doesn't help (or hurts)

- **Physics model is wrong**: subtracts the wrong thing; residual has a spurious component the detector learns as "normal."
- **Channel isn't covered**: predict() returns 0, residual = data — no benefit, no harm.
- **Anomaly looks like physics**: subtracts the anomaly along with the signal. Mitigation: physics should be insensitive to short-duration disturbances (which our Euler integrators are).

## Adding a new physics model

1. Implement the `PhysicsModel` Protocol:
   ```python
   class MyResidual:
       name = "my_model"

       def predict(self, window: TelemetryWindow) -> np.ndarray:
           ...    # return shape == window.tensor.shape

       def covered_channels(self) -> set[str]:
           return {"channel_glob_pattern"}
   ```
2. Add `from .my_model import MyResidual` to this `__init__.py`.
3. Add an entry to the `_PHYSICS_REGISTRY` dict in `__init__.py`:
   ```python
   _PHYSICS_REGISTRY["my_model"] = MyResidual
   ```
4. Configs can now reference `physics: {_target_key: my_model, ...}`.

## See also

- Phase 4 PINN bake-off config: [`configs/experiment/phase4_pinn.yaml`](../../../../../../configs/experiment/phase4_pinn.yaml)
- Layered synth components (the "ground truth" physics): [`packages/dgx_ts_lab/src/dgx_ts_lab/datasets/synthetic/layered/README.md`](../../datasets/synthetic/layered/README.md) — physics-model parameters in this dir are deliberately tuned to match the synthetic generator's defaults.
