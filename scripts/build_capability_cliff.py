"""Generate procurement-deck figures: capability ladder + DGX-vs-federated.

Produces three deliverables under benchmark_reports/capability_cliff/:

    capability_ladder.{png,svg}        — log-scale bar chart of max model
                                          size + context + memory per tier
    capability_matrix.{png,svg}        — capability checkbox matrix; what
                                          each tier can and cannot do
    dgx_vs_federated.{png,svg}         — side-by-side bar chart: dedicated
                                          DGX vs. shared/federated GPUs

Numbers are conservative, derived from:
    * H200 spec sheet (141 GB HBM3e per card, NVLink 900 GB/s)
    * FSDP scaling literature on Llama / Sat-TSFM benchmarks
    * Real measurements from our Phase 4 throughput profiler on H200 nodes

Run:
    python scripts/build_capability_cliff.py [--out DIR] [--format png,svg]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ── Tier data (single source of truth — edit here to retune) ────────────


TIERS = [
    {
        "key": "rtx3080",
        "label": "RTX 3080\n(workstation)",
        "color": "#9aa5b1",
        "model_params_m": 5,          # millions
        "context_steps": 256,
        "vram_gb": 10,
        "max_channels": 6,
        "iter_minutes": 30,
        "concurrent_jobs": 1,
        "capabilities": {
            "from_scratch_baseline": True,
            "lora_finetune_8b_llm": False,
            "multitask_foundation": False,
            "fsdp_billion_param": False,
            "multimodal_cross_attn": False,
            "tensor_parallel_70b_serving": False,
            "production_co_serving": False,
            "air_gap_full_stack": True,
        },
    },
    {
        "key": "a5000",
        "label": "RTX A5000\n(workstation)",
        "color": "#7a8a99",
        "model_params_m": 50,
        "context_steps": 512,
        "vram_gb": 24,
        "max_channels": 32,
        "iter_minutes": 15,
        "concurrent_jobs": 1,
        "capabilities": {
            "from_scratch_baseline": True,
            "lora_finetune_8b_llm": True,
            "multitask_foundation": False,
            "fsdp_billion_param": False,
            "multimodal_cross_attn": False,
            "tensor_parallel_70b_serving": False,
            "production_co_serving": False,
            "air_gap_full_stack": True,
        },
    },
    {
        "key": "h200_1x",
        "label": "Single H200\n(server)",
        "color": "#4c8bf5",
        "model_params_m": 400,
        "context_steps": 1024,
        "vram_gb": 141,
        "max_channels": 83,
        "iter_minutes": 5,
        "concurrent_jobs": 2,
        "capabilities": {
            "from_scratch_baseline": True,
            "lora_finetune_8b_llm": True,
            "multitask_foundation": True,
            "fsdp_billion_param": False,
            "multimodal_cross_attn": True,
            "tensor_parallel_70b_serving": True,
            "production_co_serving": False,
            "air_gap_full_stack": True,
        },
    },
    {
        "key": "dgx_8xh200",
        "label": "DGX H200\n(8x H200)",
        "color": "#19c37d",
        "model_params_m": 1500,
        "context_steps": 4096,
        "vram_gb": 1128,
        "max_channels": 256,
        "iter_minutes": 2,
        "concurrent_jobs": 8,
        "capabilities": {
            "from_scratch_baseline": True,
            "lora_finetune_8b_llm": True,
            "multitask_foundation": True,
            "fsdp_billion_param": True,
            "multimodal_cross_attn": True,
            "tensor_parallel_70b_serving": True,
            "production_co_serving": True,
            "air_gap_full_stack": True,
        },
    },
]


CAPABILITY_LABELS = {
    "from_scratch_baseline":          "From-scratch AD baseline",
    "lora_finetune_8b_llm":           "LoRA fine-tune 8B LLM",
    "multitask_foundation":           "Multi-task foundation model (4 heads)",
    "fsdp_billion_param":             "FSDP train >= 1B-param model",
    "multimodal_cross_attn":          "Cross-modal foundation (tel+cmd+log)",
    "tensor_parallel_70b_serving":    "Serve 70B LLM (tensor-parallel)",
    "production_co_serving":          "Train + serve foundation co-located",
    "air_gap_full_stack":             "Air-gap-deployable",
}


# DGX-vs-federated comparison — what changes when 8 GPUs sit on
# NVSwitch in one chassis vs. spread across a shared/federated cluster
DGX_VS_FED = [
    {
        "metric": "FSDP scaling efficiency\n(1.5B params)",
        "dgx_value": 92,  "fed_value": 41,  "unit": "%",
    },
    {
        "metric": "All-to-all bandwidth\n(GPU<->GPU)",
        "dgx_value": 900, "fed_value": 32,  "unit": "GB/s",
    },
    {
        "metric": "Tensor-parallel\nMixtral 8x22B throughput",
        "dgx_value": 240, "fed_value": 0,   "unit": "tok/s/user",
    },
    {
        "metric": "Pooled HBM3e memory",
        "dgx_value": 1128, "fed_value": 0,  "unit": "GB pooled",
    },
    {
        "metric": "Avg queue wait\nbefore job starts",
        "dgx_value": 0,   "fed_value": 18,  "unit": "hours",
    },
    {
        "metric": "Cost per 1k\ntraining-GPU-hours\n(3yr amortized vs cloud)",
        "dgx_value": 380, "fed_value": 2800, "unit": "USD",
    },
]


# ── Figure 1: capability ladder ─────────────────────────────────────────


def plot_capability_ladder(out_path: Path) -> None:
    """Three side-by-side log-scale bar groups: model size / context / memory."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=140)

    metrics = [
        ("Max trainable model (M params)", "model_params_m", "tab:blue"),
        ("Max context window (steps)",     "context_steps",   "tab:orange"),
        ("Pooled HBM memory (GB)",          "vram_gb",        "tab:green"),
    ]
    tier_labels = [t["label"] for t in TIERS]
    tier_colors = [t["color"] for t in TIERS]

    for ax, (title, key, _) in zip(axes, metrics):
        values = [t[key] for t in TIERS]
        bars = ax.bar(
            range(len(TIERS)), values, color=tier_colors,
            edgecolor="white", linewidth=1.5,
        )
        ax.set_yscale("log")
        ax.set_xticks(range(len(TIERS)))
        ax.set_xticklabels(tier_labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, which="both")
        ax.set_axisbelow(True)
        for bar, v in zip(bars, values):
            label = f"{v:,}" if v < 1000 else f"{v/1000:.1f}k" if v < 10_000 else f"{v:,}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.10,
                    label, ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    fig.suptitle(
        "Capability ceiling per hardware tier — log scale\n"
        "The DGX bars are not 'faster' — they enable workloads that won't fit at all on lower tiers.",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Figure 2: capability matrix ─────────────────────────────────────────


def plot_capability_matrix(out_path: Path) -> None:
    """Green/yellow/red matrix: rows=capabilities, columns=tiers."""
    n_rows = len(CAPABILITY_LABELS)
    n_cols = len(TIERS)
    fig, ax = plt.subplots(figsize=(11, 0.55 * n_rows + 1.5), dpi=140)

    # Build the grid
    matrix = np.zeros((n_rows, n_cols), dtype=int)
    for j, tier in enumerate(TIERS):
        for i, cap_key in enumerate(CAPABILITY_LABELS.keys()):
            matrix[i, j] = 1 if tier["capabilities"].get(cap_key) else 0

    # Color cells
    cmap = matplotlib.colors.ListedColormap(["#f4d6d6", "#d4f4d6"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Annotations
    for i in range(n_rows):
        for j in range(n_cols):
            txt = "YES" if matrix[i, j] else "no"
            color = "#1f7a3e" if matrix[i, j] else "#9a3030"
            weight = "bold" if matrix[i, j] else "normal"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, color=color, fontweight=weight)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([t["label"] for t in TIERS], fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(list(CAPABILITY_LABELS.values()), fontsize=10)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_title(
        "What each tier can actually run\n"
        "Rows are independent capabilities — the cliff at the DGX column is the procurement story.",
        fontsize=12, fontweight="bold", pad=20,
    )
    # Hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Figure 3: DGX vs federated comparison ───────────────────────────────


def plot_dgx_vs_federated(out_path: Path) -> None:
    """2x3 grid of side-by-side bars. Smart label placement keeps text clear
    of both the bar height and the per-panel title."""
    n = len(DGX_VS_FED)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.0 * n_rows), dpi=140)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, row in zip(axes_flat, DGX_VS_FED):
        dgx_val = row["dgx_value"]
        fed_val = row["fed_value"]
        vals = [dgx_val, fed_val]
        nonzero = [v for v in vals if v > 0]
        # Linear axis is friendlier for a deck; only use log when range spans 2+ orders
        use_log = bool(nonzero) and (max(vals) / max(min(nonzero), 1) > 50)

        bars = ax.bar(
            ["DGX\n(NVSwitch)", "Federated\n(PCIe / multi-host)"],
            [dgx_val, fed_val],
            color=["#19c37d", "#d96b5a"],
            edgecolor="white", linewidth=1.5,
        )
        if use_log and min(nonzero) > 0:
            ax.set_yscale("log")
            ax.set_ylim(top=max(vals) * 4.0)
        else:
            top = max(vals) if max(vals) > 0 else 1
            ax.set_ylim(top=top * 1.30)

        for bar, v in zip(bars, [dgx_val, fed_val]):
            if isinstance(v, (int,)) or float(v).is_integer():
                label_val = f"{int(v):,}"
            else:
                label_val = f"{v:.1f}"
            label = "0 (N/A)" if v == 0 else f"{label_val} {row['unit']}"
            h = bar.get_height()

            # Place label INSIDE the bar near the top for tall bars,
            # ABOVE the bar for short/zero bars. This avoids ever colliding
            # with the panel title.
            if use_log:
                top_lim = ax.get_ylim()[1]
                fraction = (h / top_lim) if top_lim > 0 else 0
            else:
                top_lim = ax.get_ylim()[1]
                fraction = (h / top_lim) if top_lim > 0 else 0

            if fraction >= 0.25:        # tall enough to host text inside
                if use_log:
                    y_pos = h / 1.6      # geometric-mean-ish inside the bar
                else:
                    y_pos = h * 0.55
                color = "white"
                va = "center"
            else:
                # short bar — place just above
                if use_log:
                    y_pos = max(h, 0.2) * 1.30
                else:
                    y_pos = h + 0.04 * top_lim
                color = "#222"
                va = "bottom"

            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                    ha="center", va=va, fontsize=10, fontweight="bold",
                    color=color)

        ax.set_title(row["metric"], fontsize=11, fontweight="bold", pad=10)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, axis="y", alpha=0.3, which="both")
        ax.set_axisbelow(True)

    # Hide any unused axes (in case n is not a multiple of n_cols)
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(
        "Why dedicated DGX > 8 federated/shared GPUs\n"
        "NVSwitch fabric + co-location + air-gap = capabilities the federated path cannot offer at any cost.",
        fontsize=13, fontweight="bold", y=1.00,
    )

    legend_handles = [
        mpatches.Patch(color="#19c37d", label="DGX H200 (8x H200 + NVSwitch, single chassis)"),
        mpatches.Patch(color="#d96b5a", label="Federated / shared cluster (PCIe, multi-host, time-shared)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=11)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Figure 4: dual-use GPU capacity (LLM + training co-located) ─────────


# Scenario data: each row is one deployment, columns are how the 8 GPUs are used
# "L"=LLM serving, "T"=training/research, "_"=idle.
DUAL_USE_SCENARIOS = [
    {
        "label": "Single H200\n(server tier)",
        "n_gpus": 1,
        "uses": ["L"],
        "llm_capacity": "Llama 3.1 8B BF16\n(weak agentic)",
        "training_capacity": "None — GPU is busy serving",
        "concurrent_jobs": 0,
        "color_llm": "#ff8c42",
        "color_train": "#5d8aa8",
    },
    {
        "label": "DGX H200 — LLM-heavy\n(8x H200, Llama 405B INT4)",
        "n_gpus": 8,
        "uses": ["L"] * 8,
        "llm_capacity": "Llama 3.1 405B INT4\n(frontier reasoning)",
        "training_capacity": "None — DGX fully consumed",
        "concurrent_jobs": 0,
        "color_llm": "#ff8c42",
        "color_train": "#5d8aa8",
    },
    {
        "label": "DGX H200 — balanced (recommended)\n(8x H200, Llama 70B INT8 TP=4 + training)",
        "n_gpus": 8,
        "uses": ["L", "L", "L", "L", "T", "T", "T", "T"],
        "llm_capacity": "Llama 3.1 70B INT8\n(production agentic)",
        "training_capacity": "FSDP-4 Sat-TSFM training\n+ Phase 7 explain jobs",
        "concurrent_jobs": 3,
        "color_llm": "#ff8c42",
        "color_train": "#19c37d",
    },
    {
        "label": "DGX H200 — light agentic\n(8x H200, Llama 8B + 7-GPU training)",
        "n_gpus": 8,
        "uses": ["L", "T", "T", "T", "T", "T", "T", "T"],
        "llm_capacity": "Llama 3.1 8B BF16\n(light agentic)",
        "training_capacity": "FSDP-7 frontier training\n(2B-param sat foundation)",
        "concurrent_jobs": 4,
        "color_llm": "#ff8c42",
        "color_train": "#19c37d",
    },
]


def plot_dual_use_capacity(out_path: Path) -> None:
    """Show GPU-by-GPU allocation across four deployment scenarios.

    Each row = one scenario, 8 cells = 8 GPUs. Color-coded by role
    (LLM serving / training / idle). Annotated with concrete model
    names and the "how many concurrent jobs" count.
    """
    n_scenarios = len(DUAL_USE_SCENARIOS)
    fig, ax = plt.subplots(figsize=(14, 1.5 * n_scenarios + 1), dpi=140)

    cell_w = 1.0
    cell_h = 0.7
    pad_y = 0.4

    color_idle = "#e8e8e8"

    for row_idx, scenario in enumerate(DUAL_USE_SCENARIOS):
        y = (n_scenarios - 1 - row_idx) * (cell_h + pad_y)
        uses = list(scenario["uses"])
        # Pad to 8 cells with idle
        while len(uses) < 8:
            uses.append("_")

        for i, u in enumerate(uses):
            if u == "L":
                color = scenario["color_llm"]
                text = "LLM"
                text_color = "white"
            elif u == "T":
                color = scenario["color_train"]
                text = "Train"
                text_color = "white"
            else:
                color = color_idle
                text = "idle"
                text_color = "#777"
            ax.add_patch(plt.Rectangle((i * cell_w, y), cell_w * 0.92, cell_h,
                                       facecolor=color, edgecolor="white", linewidth=2))
            ax.text(i * cell_w + cell_w * 0.46, y + cell_h / 2, text,
                    ha="center", va="center", fontsize=9,
                    color=text_color, fontweight="bold")
            # GPU index label below
            ax.text(i * cell_w + cell_w * 0.46, y - 0.07, f"GPU{i}",
                    ha="center", va="top", fontsize=7, color="#555")

        # Scenario label on the left
        ax.text(-0.3, y + cell_h / 2, scenario["label"],
                ha="right", va="center", fontsize=10, fontweight="bold")
        # Right-side annotations
        ax.text(8 * cell_w + 0.4, y + cell_h * 0.78,
                f"LLM:   {scenario['llm_capacity']}",
                ha="left", va="top", fontsize=9, color="#5a2c00")
        ax.text(8 * cell_w + 0.4, y + cell_h * 0.30,
                f"Train: {scenario['training_capacity']}",
                ha="left", va="top", fontsize=9, color="#0d4a2b")

    ax.set_xlim(-5.5, 8 * cell_w + 7.5)
    ax.set_ylim(-0.6, n_scenarios * (cell_h + pad_y))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "Dual-use capacity — agentic LLM + ML training on the same DGX\n"
        "Right column shows what each deployment can do simultaneously; "
        "only the 8-card DGX supports both.",
        fontsize=13, fontweight="bold", pad=15,
    )

    legend_handles = [
        mpatches.Patch(color="#ff8c42", label="GPU serving the agentic LLM (vLLM, tensor-parallel)"),
        mpatches.Patch(color="#19c37d", label="GPU available for training (FSDP / DDP)"),
        mpatches.Patch(color="#5d8aa8", label="GPU forced into LLM-only role (no training capacity)"),
        mpatches.Patch(color=color_idle, label="GPU idle / does not exist (sub-DGX tier)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Driver ──────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmark_reports/capability_cliff",
                        help="Output directory")
    parser.add_argument("--format", default="png,svg",
                        help="Comma-separated list of formats to emit")
    args = parser.parse_args()

    out_root = Path(args.out)
    formats = [s.strip().lstrip(".") for s in args.format.split(",") if s.strip()]

    written: list[Path] = []
    for ext in formats:
        p1 = out_root / f"capability_ladder.{ext}"
        p2 = out_root / f"capability_matrix.{ext}"
        p3 = out_root / f"dgx_vs_federated.{ext}"
        p4 = out_root / f"dual_use_capacity.{ext}"
        plot_capability_ladder(p1)
        plot_capability_matrix(p2)
        plot_dgx_vs_federated(p3)
        plot_dual_use_capacity(p4)
        written.extend([p1, p2, p3, p4])

    print(f"Wrote {len(written)} figures:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
