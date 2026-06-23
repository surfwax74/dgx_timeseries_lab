"""Detector implementations. Importing this module registers all bundled
detectors with dgx_ts_core.registry.DETECTOR_REGISTRY (and Phase 6 task
heads with HEAD_REGISTRY)."""

# heads import FIRST so HEAD_REGISTRY is populated by the time
# from_scratch.sat_tsfm_multitask wants to look up registered heads at fit().
from . import (  # noqa: F401  side-effects
    baseline,
    behavior,
    foundation,
    from_scratch,
    heads,  # noqa: F401  side-effects
    physics,
)

__all__ = ["baseline", "behavior", "foundation", "from_scratch", "heads", "physics"]
