"""Behavior-based detectors — Phase 8.

Today: operator_fingerprint (per-operator Mahalanobis distance).
Future: device fingerprint, network behavior model, etc.

Importing this module registers all bundled detectors with DETECTOR_REGISTRY.
"""

from . import operator_fingerprint  # noqa: F401  side-effect: register
from .operator_fingerprint import OperatorFingerprintDetector

__all__ = ["OperatorFingerprintDetector", "operator_fingerprint"]
