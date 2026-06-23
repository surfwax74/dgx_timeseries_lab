from __future__ import annotations

from collections.abc import Callable


class Registry[T]:
    """Simple factory registry.

    Implementations register themselves at import time via the @register
    decorator and are then instantiable by string key from Hydra configs.
    This is the mechanism behind hot-swap — `model: anomaly_transformer` in
    a YAML resolves through the registry without imports leaking into core.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if key in self._items:
                raise ValueError(f"{self._name}: '{key}' already registered")
            self._items[key] = factory
            return factory

        return decorator

    def get(self, key: str) -> Callable[..., T]:
        if key not in self._items:
            raise KeyError(
                f"{self._name}: '{key}' not registered. "
                f"Available: {sorted(self._items)}"
            )
        return self._items[key]

    def create(self, key: str, **kwargs: object) -> T:
        return self.get(key)(**kwargs)

    def list(self) -> list[str]:
        return sorted(self._items)


# Global registries — one per swappable layer of the framework.
# Implementations in dgx_ts_lab register themselves here at import time.
DATASET_REGISTRY: Registry = Registry("dataset")
DETECTOR_REGISTRY: Registry = Registry("detector")
TRAINER_REGISTRY: Registry = Registry("trainer")
# Phase 6: task heads that attach to a shared encoder (e.g., Sat-TSFM
# multi-task: fault classifier + RUL regressor + mode predictor + AD).
HEAD_REGISTRY: Registry = Registry("head")
