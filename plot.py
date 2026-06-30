"""
Generate comparison figures from experiment result CSVs.

Usage:
  python plot.py                    # reads results/raw/, saves to results/figures/
  python plot.py --dir results/raw  # explicit input directory

Figures produced:
  fig1_main_comparison.png    — EM by architecture across all four benchmark cells (k=3 baseline)
  fig2_retrieval_ablation.png — HotpotQA EM at k=3 vs k=10, with L1 full-context baseline
  fig3_token_efficiency.png   — Avg token cost by architecture across benchmark cells (k=3)
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import numpy as np
except ImportError:
    raise SystemExit("matplotlib/numpy not found — install with: pip install matplotlib numpy")


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

ARCH_ORDER = ["level1", "level2a", "level2b", "level3"]
ARCH_LABELS = {
    "level1":  "L1 Baseline",
    "level2a": "L2A Planner-Executor",
    "level2b": "L2B Solver-Critic",
    "level3":  "L3 Experience-Replay",
}
ARCH_SHORT = {
    "level1":  "L1",
    "level2a": "L2A",
    "level2b": "L2B",
    "level3":  "L3",
}
ARCH_COLORS = {
    "level1":  "#4C72B0",
    "level2a": "#DD8452",
    "level2b": "#55A868",
    "level3":  "#C44E52",
}

CELL_ORDER = [
    ("math",     "easy"),
    ("math",     "hard"),
    ("hotpotqa", "easy"),
    ("hotpotqa", "hard"),
]
CELL_LABELS = {
    ("math",     "easy"): "MATH\nEasy",
    ("math",     "hard"): "MATH\nHard",
    ("hotpotqa", "easy"): "HotpotQA\nEasy",
    ("hotpotqa", "hard"): "HotpotQA\nHard",
}

DPI = 150


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _int(v, default=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def load_and_aggregate(raw_dir: Path) -> dict:
    """
    Returns summary dict keyed by (arch, domain, diff, top_k).
    Only successful rows (no error) are included.
    top_k defaults to "3" for old CSVs that predate the column.
    """
    cells = defaultdict(lambda: {"em": [], "input_tok": [], "output_tok": []})

    for csv_path in sorted(raw_dir.glob("*.csv")):
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("error", ""):
                    continue
                arch   = row.get("architecture", "")
                domain = row.get("domain", "")
                diff   = row.get("difficulty", "")
                top_k  = row.get("top_k", "3") or "3"
                key    = (arch, domain, diff, top_k)
                cells[key]["em"].append(_int(row.get("exact_match", 0)))
                cells[key]["input_tok"].append(_int(row.get("total_input_tokens", 0)))
                cells[key]["output_tok"].append(_int(row.get("total_output_tokens", 0)))

    summary = {}
    for key, c in cells.items():
        n = len(c["em"])
        if n == 0:
            continue
        summary[key] = {
            "n":       n,
            "em":      sum(c["em"]) / n,
            "avg_tok": (sum(c["input_tok"]) + sum(c["output_tok"])) / n,
        }
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar_label(ax, bar, fmt="{:.2f}", fontsize=8, pad=0.012):
    """Place a value label just above a bar."""
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + pad,
        fmt.format(bar.get_height()),
        ha="center", va="bottom",
        fontsize=fontsize, color="0.35",
    )


# ---------------------------------------------------------------------------
# Figure 1: Main comparison (k=3 only)
# ---------------------------------------------------------------------------

def fig1_main_comparison(summary: dict, out_path: Path) -> None:
    """
    Grouped bar chart: all 4 domain×difficulty cells, 4 architectures at k=3.
    Primary result figure.
    """
    archs = [a for a in ARCH_ORDER if any(k[0] == a for k in summary)]
    n_bars = len(archs)
    bar_w = 0.18
    n_groups = len(CELL_ORDER)
    x = np.arange(n_groups) * (n_bars * bar_w + 0.28)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, arch in enumerate(archs):
        offsets = x + (i - (n_bars - 1) / 2) * bar_w
        ems = [
            summary.get((arch, dom, diff, "3"), {}).get("em")
            for dom, diff in CELL_ORDER
        ]
        for offset, em in zip(offsets, ems):
            if em is None:
                continue
            bar = ax.bar(
                offset, em,
                width=bar_w,
                color=ARCH_COLORS[arch],
                label=ARCH_LABELS[arch] if i == archs.index(arch) else "_nolegend_",
                edgecolor="white", linewidth=0.6,
            )
            _bar_label(ax, bar[0])

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in CELL_ORDER], fontsize=11)
    ax.set_ylabel("Exact Match (EM)", fontsize=12)
    ax.set_ylim(0, 1.10)
    ax.set_title("Architecture Comparison — Exact Match Accuracy  (retrieval k = 3)", fontsize=13, pad=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Retrieval ablation (HotpotQA, k=3 vs k=10)
# ---------------------------------------------------------------------------

def fig2_retrieval_ablation(summary: dict, out_path: Path) -> None:
    """
    Two subplots (easy / hard): k=3 (hatched) vs k=10 (solid) bars per architecture,
    with L1 full-context drawn as a dashed baseline.
    """
    retrieval_archs = ["level2a", "level2b", "level3"]
    diffs = ["easy", "hard"]
    diff_titles = {"easy": "HotpotQA — Easy", "hard": "HotpotQA — Hard"}

    bar_w = 0.32
    x = np.arange(len(retrieval_archs))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, diff in zip(axes, diffs):
        # L1 dashed baseline
        l1_em = summary.get(("level1", "hotpotqa", diff, "3"), {}).get("em")
        if l1_em is not None:
            ax.axhline(
                l1_em, color=ARCH_COLORS["level1"],
                linestyle="--", linewidth=2.0, zorder=5,
            )
            ax.text(
                len(retrieval_archs) - 0.05, l1_em + 0.02,
                f"L1 = {l1_em:.3f}",
                ha="right", va="bottom",
                fontsize=9, color=ARCH_COLORS["level1"],
            )

        for i, arch in enumerate(retrieval_archs):
            em_k3  = summary.get((arch, "hotpotqa", diff, "3"),  {}).get("em")
            em_k10 = summary.get((arch, "hotpotqa", diff, "10"), {}).get("em")
            color  = ARCH_COLORS[arch]

            # k=3: hatched, 55% opacity
            if em_k3 is not None:
                b3 = ax.bar(
                    x[i] - bar_w / 2, em_k3, width=bar_w,
                    color=color, alpha=0.55, hatch="///",
                    edgecolor="white", linewidth=0.5, zorder=4,
                )
                _bar_label(ax, b3[0], pad=0.010)

            # k=10: solid, full opacity
            if em_k10 is not None:
                b10 = ax.bar(
                    x[i] + bar_w / 2, em_k10, width=bar_w,
                    color=color, alpha=1.0,
                    edgecolor="white", linewidth=0.5, zorder=4,
                )
                _bar_label(ax, b10[0], pad=0.010)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [ARCH_LABELS[a] for a in retrieval_archs],
            fontsize=9,
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(diff_titles[diff], fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel("Exact Match (EM)", fontsize=12)

    # Shared legend at figure level
    legend_elements = [
        Line2D([0], [0], color=ARCH_COLORS["level1"], linestyle="--",
               linewidth=2, label="L1 Baseline (full context, no retrieval)"),
        mpatches.Patch(facecolor="0.6", hatch="///", alpha=0.55,
                       label="k = 3  (lossy retrieval)"),
        mpatches.Patch(facecolor="0.6", alpha=1.0,
                       label="k = 10  (≈ full context via retrieval)"),
    ] + [
        mpatches.Patch(facecolor=ARCH_COLORS[a], label=ARCH_LABELS[a])
        for a in retrieval_archs
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=3,
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.suptitle(
        "Retrieval Budget Ablation — HotpotQA Exact Match",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Token efficiency (k=3)
# ---------------------------------------------------------------------------

def fig3_token_efficiency(summary: dict, out_path: Path) -> None:
    """
    Grouped bar chart: avg total tokens per question by architecture (k=3 only).
    Same four-cell grouping as Figure 1.
    """
    archs = [a for a in ARCH_ORDER if any(k[0] == a for k in summary)]
    n_bars = len(archs)
    bar_w = 0.18
    n_groups = len(CELL_ORDER)
    x = np.arange(n_groups) * (n_bars * bar_w + 0.28)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, arch in enumerate(archs):
        offsets = x + (i - (n_bars - 1) / 2) * bar_w
        toks = [
            summary.get((arch, dom, diff, "3"), {}).get("avg_tok")
            for dom, diff in CELL_ORDER
        ]
        for offset, tok in zip(offsets, toks):
            if tok is None:
                continue
            bar = ax.bar(
                offset, tok,
                width=bar_w,
                color=ARCH_COLORS[arch],
                label=ARCH_LABELS[arch],
                edgecolor="white", linewidth=0.6,
            )
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                bar[0].get_height() + 30,
                f"{int(tok):,}",
                ha="center", va="bottom",
                fontsize=7, color="0.35", rotation=45,
            )

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=9, framealpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in CELL_ORDER], fontsize=11)
    ax.set_ylabel("Avg Total Tokens per Question", fontsize=12)
    ax.set_title("Token Usage by Architecture  (retrieval k = 3)", fontsize=13, pad=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate experiment figures")
    parser.add_argument("--dir", default="results/raw", help="Directory of result CSVs")
    args = parser.parse_args()

    raw_dir = Path(args.dir)
    if not raw_dir.exists():
        raise SystemExit(f"Directory not found: {raw_dir}")

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = load_and_aggregate(raw_dir)
    print(f"Loaded {len(summary)} cells.\n")

    fig1_main_comparison(summary,    figures_dir / "fig1_main_comparison.png")
    fig2_retrieval_ablation(summary, figures_dir / "fig2_retrieval_ablation.png")
    fig3_token_efficiency(summary,   figures_dir / "fig3_token_efficiency.png")

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
