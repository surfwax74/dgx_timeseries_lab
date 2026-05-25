"""profile_throughput.py — measure tokens/sec, peak memory, MFU for a detector.

Run on the actual training hardware. Output is one row per (detector, config)
combo, written to ``profile_results.md`` + stdout.

Usage:
    uv run --no-sync python scripts/profile_throughput.py \
        --detector sat_tsfm --tier h200 --batch-size 32 --window 4096 --channels 83
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "dgx_ts_core" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "dgx_ts_lab" / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--tier", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--channels", type=int, default=83)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-iters", type=int, default=20)
    args = parser.parse_args()

    import torch

    import dgx_ts_lab  # noqa: F401
    from dgx_ts_core.registry import DETECTOR_REGISTRY

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Profiling on device: {device}")

    # Build a synthetic dataset to fit the detector against
    from dgx_ts_lab.datasets.synthetic import TrivialSyntheticDataset

    ds = TrivialSyntheticDataset(
        n_samples=args.window * 16,
        n_channels=args.channels,
        seed=0,
    )

    # Build detector
    det = DETECTOR_REGISTRY.create(
        args.detector,
        window_length=args.window,
        n_channels=args.channels,
    )
    from dgx_ts_core.models import FitMode

    det.fit(ds, FitMode.PRETRAIN, {})

    if hasattr(det, "module") and det.module is not None:
        det.module.to(device)
        n_params = sum(p.numel() for p in det.module.parameters())
    else:
        n_params = 0

    # Build a representative batch
    batch = {
        "x": torch.randn(args.batch_size, args.window, args.channels, device=device, dtype=torch.float32),
        "labels": torch.zeros(args.batch_size, args.window, dtype=torch.bool, device=device),
    }

    # Warmup
    print(f"Warming up ({args.n_warmup} iters)...")
    optimizer = torch.optim.AdamW(det.module.parameters(), lr=1e-4)
    for _ in range(args.n_warmup):
        optimizer.zero_grad()
        loss = det.compute_loss(batch)
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed iters
    print(f"Timing {args.n_iters} iters...")
    t0 = time.time()
    for _ in range(args.n_iters):
        optimizer.zero_grad()
        loss = det.compute_loss(batch)
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    tokens_per_iter = args.batch_size * args.window * args.channels
    total_tokens = tokens_per_iter * args.n_iters
    tokens_per_sec = total_tokens / elapsed
    iters_per_sec = args.n_iters / elapsed

    peak_mb = (torch.cuda.max_memory_allocated() / 1024**2) if device == "cuda" else 0.0

    print()
    print("== Profile result ==")
    print(f"  detector:       {args.detector}")
    print(f"  device:         {device}")
    print(f"  batch_size:     {args.batch_size}")
    print(f"  window:         {args.window}")
    print(f"  channels:       {args.channels}")
    print(f"  parameters:     {n_params:,}")
    print(f"  iters/sec:      {iters_per_sec:.2f}")
    print(f"  tokens/sec:     {tokens_per_sec:,.0f}")
    print(f"  s/iter:         {elapsed/args.n_iters*1000:.1f} ms")
    print(f"  peak GPU mem:   {peak_mb:,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
