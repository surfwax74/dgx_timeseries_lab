"""Channel-wise z-score baseline detector.

The simplest possible AnomalyDetector — fits per-channel mean and std on
training data, then scores each step by its max channel-wise z-score.
No training loop, no GPU needed, fully deterministic. Used by the Phase 0
smoke test to prove the scaffold runs end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dgx_ts_core.data import TelemetryDataset, TelemetryWindow
from dgx_ts_core.models import (
    AnomalyScore,
    Capabilities,
    FitMode,
    FitResult,
    OutputKind,
)
from dgx_ts_core.registry import DETECTOR_REGISTRY


class RollingMeanDetector:
    """Per-channel rolling-mean residual detector.

    Fit: compute per-channel σ from training data (sets the score scale).
    Score: for each step t, compute a causal rolling mean of size W ending
    at t, then score = max_c |x[t,c] − rolling_mean[t,c]| / σ_c.

    A spike on top of a smooth signal produces a large residual against
    the rolling mean — making this baseline actually useful for the
    point-anomaly case the smoke test exercises.
    """

    def __init__(self, window_size: int = 32) -> None:
        self._window_size = window_size
        self._stds: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "rolling_mean"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            requires_pretraining=False,
            supports_streaming=True,
            supports_multivariate=True,
            native_context_len=10_000,
            output_kind=OutputKind.PER_STEP,
            supports_peft=False,
            supports_export_onnx=False,
            supports_zero_shot=True,
        )

    @property
    def _fitted(self) -> bool:
        return self._stds is not None

    def fit(
        self,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: dict[str, Any],
    ) -> FitResult:
        # Walk training data once, compute the natural variation of the
        # rolling-mean residual per channel. This is the scale that makes
        # a residual "1σ" mean "typical noise" — much sharper than using
        # value std, which would over-weight signal curvature.
        chunk_len = int(config.get("calibration_chunk_len", 1024))
        all_residuals: list[np.ndarray] = []
        n_steps = 0
        for window in dataset.windows(length=chunk_len, stride=chunk_len):
            rolling = _causal_rolling_mean(window.tensor, self._window_size)
            all_residuals.append(window.tensor - rolling)
            n_steps += window.length
        if all_residuals:
            residuals = np.concatenate(all_residuals, axis=0)
            self._stds = (residuals.std(axis=0) + 1e-8).astype(np.float32)
        else:
            # Dataset shorter than chunk_len → fall back to value std.
            stats = dataset.stats()
            self._stds = stats.stds.copy()
            n_steps = stats.n_samples
        return FitResult(
            detector_name=self.name,
            mode=mode,
            n_steps=n_steps,
            metadata={
                "residual_stds": self._stds.tolist(),
                "window_size": self._window_size,
            },
        )

    def score(self, window: TelemetryWindow) -> AnomalyScore:
        if not self._fitted:
            raise RuntimeError("RollingMeanDetector must be fit before scoring")
        assert self._stds is not None
        rolling = _causal_rolling_mean(window.tensor, self._window_size)
        z = np.abs(window.tensor - rolling) / self._stds  # (T, C)
        per_step = z.max(axis=1).astype(np.float32)        # (T,)
        return AnomalyScore(scores=per_step)

    def embed(self, window: TelemetryWindow) -> Any:
        raise NotImplementedError("RollingMeanDetector has no latent embedding")

    def reconstruct(self, window: TelemetryWindow) -> Any:
        raise NotImplementedError("RollingMeanDetector does not reconstruct")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._fitted:
            raise RuntimeError("nothing to save — detector not fit")
        assert self._stds is not None
        np.savez(path, stds=self._stds, window_size=np.array([self._window_size]))

    @classmethod
    def load(cls, path: Path) -> "RollingMeanDetector":
        data = np.load(Path(path))
        ws = int(data["window_size"][0]) if "window_size" in data else 32
        det = cls(window_size=ws)
        det._stds = data["stds"]
        return det


def _causal_rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Backward-looking rolling mean. For t < w, uses the cumulative mean.

    Shapes:  x: (T, C)  →  out: (T, C)
    Vectorized via cumulative sums; O(T·C).
    """
    t, c = x.shape
    if w <= 1 or t <= 1:
        return x.copy()
    w_eff = min(w, t)
    cumsum = np.concatenate(
        [np.zeros((1, c), dtype=np.float32), x.cumsum(axis=0, dtype=np.float32)],
        axis=0,
    )  # (T+1, C)
    out = np.empty((t, c), dtype=np.float32)
    # Warm-up: cumulative mean for the first w_eff−1 steps.
    counts = np.arange(1, w_eff, dtype=np.float32)[:, None]
    out[: w_eff - 1] = cumsum[1:w_eff] / counts
    # Steady state: fixed-window mean of size w_eff.
    out[w_eff - 1 :] = (cumsum[w_eff:] - cumsum[: t - w_eff + 1]) / w_eff
    return out


@DETECTOR_REGISTRY.register("rolling_mean")
def _create(**kwargs: object) -> RollingMeanDetector:
    return RollingMeanDetector(**kwargs)  # type: ignore[arg-type]
