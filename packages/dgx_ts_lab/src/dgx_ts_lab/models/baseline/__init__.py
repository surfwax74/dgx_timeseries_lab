from . import rolling_mean  # noqa: F401  side-effect: register rolling_mean
from .rolling_mean import RollingMeanDetector

__all__ = ["RollingMeanDetector", "rolling_mean"]
