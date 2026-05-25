# Adding a new dataset

Five steps. None require touching `dgx_ts_core` (unless you're changing the Protocol itself).

## 1. Implement the Protocol

Subclass nothing — just match the shape. The Protocol is in [`packages/dgx_ts_core/src/dgx_ts_core/data/dataset.py`](../packages/dgx_ts_core/src/dgx_ts_core/data/dataset.py).

```python
# packages/dgx_ts_lab/src/dgx_ts_lab/datasets/my_source.py
from collections.abc import Iterator, Mapping
import numpy as np

from dgx_ts_core.data import (
    Channel, DatasetStats, SplitScheme, Subsystem, TelemetryWindow, Units,
)

class MySourceDataset:
    @property
    def name(self) -> str: ...
    @property
    def subsystem(self) -> Subsystem: ...
    @property
    def channels(self) -> tuple[Channel, ...]: ...
    @property
    def sample_rate_hz(self) -> float: ...
    @property
    def has_labels(self) -> bool: ...

    def windows(self, length: int, stride: int) -> Iterator[TelemetryWindow]: ...
    def split(self, scheme: SplitScheme) -> Mapping[str, "MySourceDataset"]: ...
    def stats(self) -> DatasetStats: ...
```

## 2. Register a factory

```python
from dgx_ts_core.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register("my_source")
def _create(**kwargs) -> MySourceDataset:
    return MySourceDataset(**kwargs)
```

## 3. Add it to the side-effect import chain

In `packages/dgx_ts_lab/src/dgx_ts_lab/datasets/__init__.py`:

```python
from . import my_source  # noqa: F401  side-effect: register my_source
```

This makes `import dgx_ts_lab` trigger your registration.

## 4. Add a Hydra config

```yaml
# configs/dataset/my_source.yaml
_target_key: my_source

# your kwargs:
my_param: 42
data_root: data/my_source
```

## 5. Write tests + update READMEs

- Add `packages/dgx_ts_lab/tests/test_my_source.py` covering construction, windows shape, split correctness, determinism.
- Update [`packages/dgx_ts_lab/src/dgx_ts_lab/datasets/README.md`](../packages/dgx_ts_lab/src/dgx_ts_lab/datasets/README.md) to list the new file + registry key.
- Update [`configs/dataset/README.md`](../configs/dataset/README.md) to list the new YAML.

## Verify

```powershell
uv run pytest packages/dgx_ts_lab/tests/test_my_source.py -v
uv run dgx-ts train dataset=my_source model=rolling_mean trainer=single_cpu mode=zeroshot
```

## Air-gap considerations

- If your dataset loads files from disk, never auto-download. Check that files exist, fail loudly with the expected layout and source URL if missing.
- See [`air_gapped_setup.md`](air_gapped_setup.md) for how data dirs are provisioned on the DGX.
