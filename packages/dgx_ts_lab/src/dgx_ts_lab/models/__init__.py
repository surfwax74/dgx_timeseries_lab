"""Detector implementations. Importing this module registers all bundled
detectors with dgx_ts_core.registry.DETECTOR_REGISTRY."""

from . import baseline, foundation, from_scratch, physics  # noqa: F401  side-effects

__all__ = ["baseline", "foundation", "from_scratch", "physics"]
