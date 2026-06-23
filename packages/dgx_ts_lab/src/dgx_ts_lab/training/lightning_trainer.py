"""Lightning Fabric-based Trainer implementation.

Branches on detector.capabilities to choose between:
  - the lightweight path: detector.fit() handles everything (used by classical
    baselines like RollingMeanDetector that have no gradient training);
  - the Fabric loop: full distributed training (added in Phase 2 for neural
    detectors like Anomaly Transformer, DCdetector, PatchTST+MAE).

The Phase 0 acceptance test exercises only the lightweight path. The
Fabric path is a clearly marked stub.
"""

from __future__ import annotations

import numpy as np
from dgx_ts_core.data import SplitScheme, SplitStrategy, TelemetryDataset
from dgx_ts_core.evaluation.result import EvalReport
from dgx_ts_core.models import AnomalyDetector, FitMode, FitResult
from dgx_ts_core.registry import TRAINER_REGISTRY
from dgx_ts_core.training import TrainConfig

from ..evaluation import basic_metrics, calibrate_threshold


class LightningTrainer:
    """Trainer Protocol implementation.

    Phase 0: drives the classical-detector path (fit + per-window scoring +
    percentile-based threshold calibration + val/test eval).

    Phase 2: adds the Lightning Fabric loop for neural detectors with
    configurable strategy (single_gpu | ddp | fsdp | deepspeed_*).
    """

    def fit(
        self,
        detector: AnomalyDetector,
        dataset: TelemetryDataset,
        mode: FitMode,
        config: TrainConfig,
    ) -> FitResult:
        splits = dataset.split(
            SplitScheme(
                strategy=SplitStrategy.TEMPORAL,
                train_frac=0.7,
                val_frac=0.15,
                test_frac=0.15,
                seed=config.seed,
            )
        )
        train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]

        caps = detector.capabilities
        needs_neural_loop = caps.requires_pretraining and mode != FitMode.ZEROSHOT

        if needs_neural_loop:
            # Run the Fabric loop, then run our standard scoring/calibration
            # pipeline so the same val/test metrics surface as for classical detectors.
            fit_result = self._fabric_fit(detector, train_ds, val_ds, mode, config)
            train_scores = _score_dataset(
                detector, train_ds, config.window_length, config.window_stride
            )
            threshold = calibrate_threshold(
                train_scores["scores"], method="percentile", percentile=99.0
            )
            val_metrics: dict[str, float] = {}
            test_metrics: dict[str, float] = {}
            val_arrays: dict[str, np.ndarray] = {}
            test_arrays: dict[str, np.ndarray] = {}
            if val_ds.has_labels:
                v = _score_dataset(
                    detector, val_ds, config.window_length, config.window_stride
                )
                val_metrics = basic_metrics(v["labels"], v["scores"], threshold)
                val_arrays = v
            if test_ds.has_labels:
                te = _score_dataset(
                    detector, test_ds, config.window_length, config.window_stride
                )
                test_metrics = basic_metrics(te["labels"], te["scores"], threshold)
                test_arrays = te
            return FitResult(
                detector_name=fit_result.detector_name,
                mode=fit_result.mode,
                final_loss=fit_result.final_loss,
                n_steps=fit_result.n_steps,
                artifacts=fit_result.artifacts,
                metadata={
                    **fit_result.metadata,
                    "threshold": threshold,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "val_arrays": val_arrays,
                    "test_arrays": test_arrays,
                },
            )

        # Lightweight path — let the detector do whatever it does internally.
        fit_result = detector.fit(
            train_ds, mode, {"window_length": config.window_length}
        )

        # Calibrate threshold on train scores (unsupervised, percentile-based).
        train_scores = _score_dataset(detector, train_ds, config.window_length, config.window_stride)
        threshold = calibrate_threshold(train_scores["scores"], method="percentile", percentile=99.0)

        # Eval on val + test if labels are available.
        val_metrics: dict[str, float] = {}
        test_metrics: dict[str, float] = {}
        val_arrays: dict[str, np.ndarray] = {}
        test_arrays: dict[str, np.ndarray] = {}
        if val_ds.has_labels:
            v = _score_dataset(detector, val_ds, config.window_length, config.window_stride)
            val_metrics = basic_metrics(v["labels"], v["scores"], threshold)
            val_arrays = v
        if test_ds.has_labels:
            te = _score_dataset(detector, test_ds, config.window_length, config.window_stride)
            test_metrics = basic_metrics(te["labels"], te["scores"], threshold)
            test_arrays = te

        return FitResult(
            detector_name=fit_result.detector_name,
            mode=fit_result.mode,
            final_loss=fit_result.final_loss,
            n_steps=fit_result.n_steps,
            artifacts=fit_result.artifacts,
            metadata={
                **fit_result.metadata,
                "threshold": threshold,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "val_arrays": val_arrays,
                "test_arrays": test_arrays,
            },
        )

    def zero_shot(
        self,
        detector: AnomalyDetector,
        dataset: TelemetryDataset,
    ) -> EvalReport:
        # Score the full dataset, calibrate percentile threshold on the same
        # split (since zero-shot has no training data to calibrate on),
        # report metrics if labels are available.
        scored = _score_dataset(detector, dataset, length=256, stride=256)
        threshold = calibrate_threshold(scored["scores"], method="percentile", percentile=99.0)
        metrics: dict[str, float] = {}
        if dataset.has_labels:
            metrics = basic_metrics(scored["labels"], scored["scores"], threshold)
        return EvalReport(
            detector_name=detector.name,
            dataset_name=dataset.name,
            metrics=metrics,
            threshold=threshold,
        )

    def _fabric_fit(
        self,
        detector: AnomalyDetector,
        train_dataset: TelemetryDataset,
        val_dataset: TelemetryDataset,
        mode: FitMode,
        config: TrainConfig,
    ) -> FitResult:
        # Phase 2 implementation: delegate to the Fabric loop module.
        from .fabric_loop import fabric_fit

        return fabric_fit(detector, train_dataset, val_dataset, mode, config)


def _score_dataset(
    detector: AnomalyDetector,
    dataset: TelemetryDataset,
    length: int,
    stride: int,
) -> dict[str, np.ndarray]:
    """Run detector over every window and concatenate scores (+ labels)."""
    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for window in dataset.windows(length=length, stride=stride):
        s = detector.score(window)
        score_chunks.append(np.asarray(s.scores, dtype=np.float32))
        if window.labels is not None:
            label_chunks.append(np.asarray(window.labels, dtype=np.bool_))
    if not score_chunks:
        return {
            "scores": np.zeros(0, dtype=np.float32),
            "labels": np.zeros(0, dtype=np.bool_),
        }
    return {
        "scores": np.concatenate(score_chunks),
        "labels": (
            np.concatenate(label_chunks) if label_chunks else np.zeros(0, dtype=np.bool_)
        ),
    }


@TRAINER_REGISTRY.register("lightning")
def _create(**kwargs: object) -> LightningTrainer:
    return LightningTrainer()
