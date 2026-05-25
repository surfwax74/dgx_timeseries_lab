"""Lightning Fabric training loop for neural detectors.

The detector ↔ trainer contract (locked Phase 2):

    A "neural" detector (``capabilities.requires_pretraining == True``)
    must expose:
        - ``module: nn.Module``         the trainable parameters
        - ``compute_loss(batch) -> Tensor``     for the training step
        - ``compute_score_batch(batch) -> Tensor``    for per-step AD scores

This loop owns Fabric setup, optimizer construction, the train/val steps,
gradient clipping, per-epoch checkpointing (best-val-loss), and prints.
Distributed strategy is selected from ``config.strategy`` so swapping from
single-GPU to FSDP in Phase 4 is a config change, not a code change.
"""

from __future__ import annotations

import time
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from dgx_ts_core.data import TelemetryDataset
from dgx_ts_core.models import FitMode, FitResult
from dgx_ts_core.training import TrainConfig

from .window_dataset import WindowTorchDataset


def _resolve_accelerator(device: str) -> str:
    if device == "auto":
        return "auto"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def fabric_fit(
    detector,
    train_dataset: TelemetryDataset,
    val_dataset: TelemetryDataset,
    mode: FitMode,
    config: TrainConfig,
) -> FitResult:
    """Train ``detector`` on ``train_dataset`` with Fabric.

    Returns a FitResult with loss history in metadata + best checkpoint path
    in artifacts. The Trainer's higher-level fit() handles MLflow logging
    and val/test metric calculation against the trained detector.
    """
    # ── pre-flight: verify the detector exposes the neural-detector surface ──
    for attr in ("module", "compute_loss", "compute_score_batch"):
        if not hasattr(detector, attr):
            raise AttributeError(
                f"detector {detector.name!r} declares requires_pretraining=True "
                f"but is missing .{attr}. Neural detectors must expose "
                "module, compute_loss(batch), and compute_score_batch(batch)."
            )

    # Build the module + populate normalization buffers if the detector hasn't
    # been initialized yet. detector.fit() is idempotent in our convention.
    detector.fit(train_dataset, mode, {"window_length": config.window_length})

    # If FSDP is requested, build the strategy kwargs from config.extra.
    strategy: str | object = config.strategy
    if config.strategy == "fsdp":
        try:
            from .strategies.fsdp import build_fsdp_strategy_kwargs
            from lightning.fabric.strategies import FSDPStrategy

            fsdp_kwargs = build_fsdp_strategy_kwargs(dict(config.extra))
            strategy = FSDPStrategy(**{k: v for k, v in fsdp_kwargs.items() if v is not None})
        except Exception as e:  # noqa: BLE001
            # Fall back to the string strategy; Fabric will use defaults.
            print(f"NOTE: FSDP strategy kwargs unavailable ({e}); using Fabric defaults.")

    fabric = L.Fabric(
        accelerator=_resolve_accelerator(config.device),
        devices="auto",
        strategy=strategy,
        precision=config.precision,
    )
    fabric.launch()

    # ── dataloaders ────────────────────────────────────────────────────
    train_torch = WindowTorchDataset(
        train_dataset, length=config.window_length, stride=config.window_stride
    )
    val_torch = WindowTorchDataset(
        val_dataset, length=config.window_length, stride=config.window_length
    )
    train_loader = DataLoader(
        train_torch,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_torch,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    train_loader = fabric.setup_dataloaders(train_loader)
    val_loader = fabric.setup_dataloaders(val_loader)

    # ── module + optimizer ─────────────────────────────────────────────
    module = detector.module
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=config.learning_rate,
        weight_decay=float(config.extra.get("weight_decay", 0.0)),
    )
    module, optimizer = fabric.setup(module, optimizer)
    # In case Fabric wrapped the module, write it back so the detector
    # picks up the wrapped reference for compute_loss/compute_score_batch.
    detector.module = module

    # ── loop ───────────────────────────────────────────────────────────
    grad_clip = float(config.extra.get("grad_clip_norm", 1.0))
    log_every = int(config.extra.get("log_every_n_steps", 50))
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = config.checkpoint_dir / f"{detector.name}_best.pt"

    train_history: list[float] = []
    val_history: list[float] = []
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(config.max_epochs):
        module.train()
        epoch_train: list[float] = []
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = detector.compute_loss(batch)
            fabric.backward(loss)
            if grad_clip > 0:
                fabric.clip_gradients(module, optimizer, max_norm=grad_clip)
            optimizer.step()
            epoch_train.append(float(loss.detach().cpu()))
            global_step += 1
            if step % log_every == 0:
                fabric.print(
                    f"  epoch {epoch:3d} step {step:5d} train_loss={loss.item():.4f}"
                )
        avg_train = sum(epoch_train) / max(1, len(epoch_train))

        module.eval()
        epoch_val: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                v = detector.compute_loss(batch)
                epoch_val.append(float(v.detach().cpu()))
        avg_val = sum(epoch_val) / max(1, len(epoch_val)) if epoch_val else float("nan")

        train_history.append(avg_train)
        val_history.append(avg_val)
        elapsed = time.time() - t0
        fabric.print(
            f"epoch {epoch:3d} | train={avg_train:.4f} | val={avg_val:.4f} | {elapsed:.1f}s"
        )

        if avg_val == avg_val and avg_val < best_val_loss:  # not NaN
            best_val_loss = avg_val
            fabric.save(
                best_ckpt,
                {
                    "module_state": module.state_dict(),
                    "epoch": epoch,
                    "val_loss": avg_val,
                },
            )

    return FitResult(
        detector_name=detector.name,
        mode=mode,
        final_loss=val_history[-1] if val_history else None,
        n_steps=global_step,
        artifacts={"best_checkpoint": best_ckpt},
        metadata={
            "train_loss_history": train_history,
            "val_loss_history": val_history,
            "best_val_loss": best_val_loss,
            "epochs": config.max_epochs,
        },
    )
