"""Critical invariant: dgx_ts_core must be importable without torch.

This is the lift-to-MLOps contract — downstream systems can `pip install
dgx-ts-core` and consume model_card / feature_schema artifacts without
dragging training dependencies into their runtime.
"""

from __future__ import annotations

import importlib
import sys


def test_core_imports_without_torch() -> None:
    # If something accidentally imports torch (or lightning, mlflow, hydra),
    # this test catches it at CI time.
    forbidden = {"torch", "lightning", "pytorch_lightning", "mlflow", "hydra"}

    # Take a snapshot of what's already imported (e.g. by other tests).
    already_loaded = forbidden & set(sys.modules)

    # Re-import the core modules cleanly.
    for mod_name in [
        "dgx_ts_core",
        "dgx_ts_core.data",
        "dgx_ts_core.models",
        "dgx_ts_core.training",
        "dgx_ts_core.evaluation",
        "dgx_ts_core.export",
        "dgx_ts_core.registry",
    ]:
        importlib.import_module(mod_name)

    leaked = (forbidden & set(sys.modules)) - already_loaded
    assert not leaked, f"dgx_ts_core leaked forbidden imports: {sorted(leaked)}"
