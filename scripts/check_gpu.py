"""GPU preflight check — what hardware do we have, what tier does it fit?

Used by the per-tier launch scripts to fail-fast with a clear message if
the box can't run what the trainer config asks for.

Exit codes:
    0  preflight OK
    1  CUDA not available (or torch was installed CPU-only)
    2  too few GPUs
    3  insufficient VRAM
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass
class GpuInfo:
    index: int
    name: str
    total_vram_gb: float
    free_vram_gb: float
    compute_capability: tuple[int, int]


def query_gpus() -> list[GpuInfo]:
    try:
        import torch
    except ImportError:
        print("ERROR: torch is not installed. uv sync should pull it.", file=sys.stderr)
        sys.exit(1)
    if not torch.cuda.is_available():
        return []
    gpus: list[GpuInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        gpus.append(
            GpuInfo(
                index=i,
                name=props.name,
                total_vram_gb=total / 1024**3,
                free_vram_gb=free / 1024**3,
                compute_capability=(props.major, props.minor),
            )
        )
    return gpus


def recommend_tier(gpus: list[GpuInfo]) -> str:
    if not gpus:
        return "cpu"
    n = len(gpus)
    vram = max(g.total_vram_gb for g in gpus)
    name_l = " ".join(g.name.lower() for g in gpus)

    if "h100" in name_l or "h200" in name_l:
        return "h200_fsdp_8x" if n >= 4 else "h200"
    if "a100" in name_l:
        return "a100_fsdp_8x" if n >= 4 else "a100"
    if "a5000" in name_l or "a6000" in name_l:
        return "a5000_x8" if n >= 4 else "a5000"
    if "rtx 3080" in name_l or "rtx 4080" in name_l or "rtx 3090" in name_l or "rtx 4090" in name_l:
        return "rtx3080_x2" if n >= 2 else "rtx3080"
    # Fallback by VRAM
    if vram >= 80:
        return "h200_fsdp_8x" if n >= 4 else "h200"
    if vram >= 40:
        return "a5000_x8" if n >= 4 else "a5000"
    if vram >= 16:
        return "rtx3080_x2" if n >= 2 else "rtx3080"
    if vram >= 8:
        return "rtx3080"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require", type=int, default=0, help="Minimum GPU count required")
    parser.add_argument("--min-vram-gb", type=float, default=0.0, help="Minimum VRAM per GPU required")
    parser.add_argument("--recommended-tier", type=str, default=None,
                        help="Warn if detected tier doesn't match this one")
    parser.add_argument("--quiet", action="store_true", help="Print only on error")
    args = parser.parse_args()

    gpus = query_gpus()
    tier = recommend_tier(gpus)

    if not args.quiet:
        print(f"== GPU preflight ==")
        if not gpus:
            print("  No CUDA GPUs detected (CPU-only mode).")
            print("  - On Windows, install the CUDA-enabled torch wheel:")
            print("      uv pip install torch --index-url https://download.pytorch.org/whl/cu124")
            print("    (or cu126 / cu128 for newer drivers)")
            print("  - Verify your NVIDIA driver: nvidia-smi")
        else:
            for g in gpus:
                print(
                    f"  [{g.index}] {g.name:<30s} "
                    f"VRAM {g.total_vram_gb:5.1f} GB total ({g.free_vram_gb:5.1f} GB free) "
                    f"cc {g.compute_capability[0]}.{g.compute_capability[1]}"
                )
        print(f"  Recommended trainer config: trainer={tier}")
        print()

    # Hard checks
    if args.require > len(gpus):
        print(
            f"FAIL: need {args.require} GPU(s), found {len(gpus)}.",
            file=sys.stderr,
        )
        return 2
    if args.min_vram_gb > 0 and gpus:
        smallest = min(g.total_vram_gb for g in gpus)
        if smallest < args.min_vram_gb:
            print(
                f"FAIL: need {args.min_vram_gb:.1f} GB VRAM, smallest GPU has {smallest:.1f} GB.",
                file=sys.stderr,
            )
            return 3
    if args.recommended_tier and args.recommended_tier != tier:
        print(
            f"NOTE: requested tier '{args.recommended_tier}' but detected '{tier}'. "
            "Continuing — you can override with the chosen tier.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
