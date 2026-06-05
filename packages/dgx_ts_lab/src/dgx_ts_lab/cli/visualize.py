"""``dgx-ts viz`` — render ROC / PR / AUC plots from a benchmark output dir.

Usage:
    dgx-ts viz --benchmark-dir benchmark_reports/phase2_bakeoff
              [--output-dir benchmark_reports/phase2_bakeoff/figures]
              [--format png,svg]
              [--splits val,test]

Output: one ROC overlay + one PR overlay per (dataset, split), plus a
single AUC bar chart over all detectors on val. Drop the resulting PNGs
straight into slides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..evaluation.visualize import render_benchmark_report_figures


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dgx-ts viz",
        description="Render benchmark figures (ROC / PR / AUC bars).",
    )
    p.add_argument(
        "--benchmark-dir",
        required=True,
        help="Directory containing benchmark_report.json + per-run *.npz",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write figures (default: <benchmark-dir>/figures)",
    )
    p.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated list of splits to plot (val,test)",
    )
    p.add_argument(
        "--format",
        default="png",
        help="Comma-separated list of file formats (png,svg,pdf)",
    )
    return p


def run() -> None:
    args = _build_argparser().parse_args()
    bd = Path(args.benchmark_dir)
    out = Path(args.output_dir) if args.output_dir else bd / "figures"
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    formats = tuple(s.strip().lstrip(".") for s in args.format.split(",") if s.strip())

    print(f"-- Rendering benchmark figures")
    print(f"   from:    {bd}")
    print(f"   to:      {out}")
    print(f"   splits:  {splits}")
    print(f"   formats: {formats}")

    written = render_benchmark_report_figures(
        benchmark_dir=bd,
        output_dir=out,
        splits=splits,
        formats=formats,
    )
    if not written:
        print("WARN: no figures rendered (no per-run .npz files found?)")
        return
    print(f"-- Wrote {len(written)} figure(s):")
    for p in written:
        print(f"   {p}")


if __name__ == "__main__":
    run()
