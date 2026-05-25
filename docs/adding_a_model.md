# Adding a new detector

Six steps. As with datasets, you don't touch `dgx_ts_core` unless changing the Protocol itself.

## 1. Implement the Protocol

The Protocol is in [`packages/dgx_ts_core/src/dgx_ts_core/models/detector.py`](../packages/dgx_ts_core/src/dgx_ts_core/models/detector.py).

```python
# packages/dgx_ts_lab/src/dgx_ts_lab/models/<bucket>/my_detector.py
from pathlib import Path
from typing import Any

from dgx_ts_core.data import TelemetryDataset, TelemetryWindow
from dgx_ts_core.models import (
    AnomalyScore, Capabilities, FitMode, FitResult, OutputKind,
)

class MyDetector:
    @property
    def name(self) -> str: return "my_detector"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(...)

    def fit(self, dataset: TelemetryDataset, mode: FitMode, config: dict[str, Any]) -> FitResult: ...
    def score(self, window: TelemetryWindow) -> AnomalyScore: ...
    def embed(self, window): raise NotImplementedError
    def reconstruct(self, window): raise NotImplementedError
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "MyDetector": ...
```

## 2. Declare honest Capabilities

Trainers branch on these, so lying breaks the bake-off:

```python
Capabilities(
    requires_pretraining=True,      # needs the Lightning Fabric loop
    supports_streaming=False,
    supports_multivariate=True,
    native_context_len=4096,
    output_kind=OutputKind.PER_STEP,
    supports_peft=True,             # LoRA / adapter compatible
    supports_export_onnx=True,
    supports_zero_shot=False,
)
```

## 3. Register

```python
from dgx_ts_core.registry import DETECTOR_REGISTRY

@DETECTOR_REGISTRY.register("my_detector")
def _create(**kwargs) -> MyDetector:
    return MyDetector(**kwargs)
```

## 4. Wire side-effect import

Add `from . import my_detector` (or `from . import my_bucket`) to the relevant `__init__.py` so importing `dgx_ts_lab.models` registers it.

## 5. Config

```yaml
# configs/model/my_detector.yaml
_target_key: my_detector

# your hyperparameters
d_model: 512
n_layers: 6
```

## 6. Tests + READMEs

- Tests should cover: protocol conformance (`isinstance(det, AnomalyDetector)`), capabilities declaration, fit → score round trip, save → load round trip, ROC-AUC on a known easy dataset.
- Update [`packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/models/README.md).
- Update [`configs/model/README.md`](../configs/model/README.md).

## Verify

```powershell
uv run pytest packages/dgx_ts_lab/tests/test_my_detector.py -v
uv run dgx-ts train model=my_detector dataset=trivial_synth trainer=single_cpu
```

## If your detector needs gradient training

Today's `LightningTrainer._fabric_fit` is a stub raising `NotImplementedError`. Phase 2 fills it in. Until then, your detector either needs `requires_pretraining=False` (do training inside `.fit()` yourself) or you'll need to land the Fabric loop alongside your detector — see the Phase 2 plan.
