#!/usr/bin/env python3
"""
Plot scaling results produced by parse_runs.py.

Usage:
    python3 plot.py results.csv
    python3 plot.py results.csv --outdir some/other/dir
    python3 plot.py results.csv --sizes 100x100x100x100,475x475x475x475
    python3 plot.py results.csv --no-computation

Input is the CSV emitted by `parse_runs.py <run_dir> --csv [--debug]`:
    file,m1,n1,m2,n2,ranks,op,alg,all_gather,computation,reduce_scatter,total,failed
optionally followed by a "# per_rank" section with per-rank timings.

Plots are written to <csv_dir>/plots/ unless --outdir is given.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.patches import Patch

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#4a4a4a",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
})

COMPONENT_COLORS = {
    "all_gather": "#4C72B0",
    "reduce_scatter": "#DD8452",
    "computation": "#55A868",
}
COMPONENT_LABELS = {
    "all_gather": "All Gather",
    "reduce_scatter": "Reduce Scatter",
    "computation": "Computation",
}
HATCHES = [None, "//", "\\\\", "xx", "..", "++"]

ALG_LABELS = {"wbp": "WBP", "rrp": "RRP", "bcp": "BCP"}


def alg_label(alg):
    return ALG_LABELS.get(alg, alg.upper())


def time_formatter(x, _pos=None):
    if x <= 0:
        return "0"
    if x >= 1:
        val, unit = x, "s"
    elif x >= 1e-3:
        val, unit = x * 1e3, "ms"
    else:
        val, unit = x * 1e6, "µs"
    s = f"{val:.1f}".rstrip("0").rstrip(".")
    return f"{s}{unit}"


# ── Loading ──────────────────────────────────────────────────────────────────

def load_csv(path):
    """Load the mean-timings section and, if present, the per-rank section."""
    path = Path(path)
    lines = path.read_text().splitlines()

    try:
        split_idx = next(i for i, l in enumerate(lines) if l.strip() == "# per_rank")
    except StopIteration:
        split_idx = None

    from io import StringIO

    mean_text = "\n".join(lines[:split_idx] if split_idx is not None else lines)
    df = pd.read_csv(StringIO(mean_text))
    for col in ("all_gather", "computation", "reduce_scatter", "total"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    per_rank = None
    if split_idx is not None:
        rank_text = "\n".join(lines[split_idx + 1:])
        rank_text = "\n".join(l for l in rank_text.splitlines() if l.strip())
        if rank_text.strip():
            per_rank = pd.read_csv(StringIO(rank_text))
            for col in ("all_gather", "computation", "reduce_scatter"):
                per_rank[col] = pd.to_numeric(per_rank[col], errors="coerce")

    return df, per_rank


def parse_size_filter(spec):
    """Parse '100x100x100x100,200x200x1000x1000' into a set of (m1,n1,m2,n2) tuples."""
    sizes = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = tuple(int(x) for x in chunk.lower().split("x"))
        if len(parts) != 4:
            sys.exit(f"Error: invalid size '{chunk}', expected form MxNxMxN")
        sizes.add(parts)
    return sizes


# ── Strong scaling grid ──────────────────────────────────────────────────────

def strong_scaling_plot(df, outdir, sizes=None, components=("all_gather", "reduce_scatter", "computation"),
                         out_name="strong_scaling", title=None):
    components = [c for c in components if c in COMPONENT_COLORS]

    ops = [op for op in ("Ax", "ATx") if op in df["op"].unique()]
    if not ops:
        ops = sorted(df["op"].unique())

    all_sizes = sorted(df[["m1", "n1", "m2", "n2"]].drop_duplicates().itertuples(index=False, name=None))
    if sizes:
        all_sizes = [s for s in all_sizes if s in sizes]
    if not all_sizes:
        print("No matching problem sizes found; skipping strong scaling plot.")
        return

    algs = sorted(df["alg"].unique())
    hatch_for = {a: HATCHES[i % len(HATCHES)] for i, a in enumerate(algs)}

    nrows, ncols = len(ops), len(all_sizes)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(4.6 * ncols, 7.5), 4.2 * nrows),
        squeeze=False,
    )

    for col, size in enumerate(all_sizes):
        m1, n1, m2, n2 = size
        df_size = df[(df["m1"] == m1) & (df["n1"] == n1) & (df["m2"] == m2) & (df["n2"] == n2)]

        for row, op in enumerate(ops):
            ax = axes[row][col]
            df_op = df_size[df_size["op"] == op]

            if df_op.empty:
                ax.axis("off")
                continue

            agg = (
                df_op.groupby(["alg", "ranks"], as_index=False)[["all_gather", "reduce_scatter", "computation"]]
                .mean()
            )

            all_ranks = sorted(agg["ranks"].unique())
            x = np.arange(len(all_ranks))
            width = 0.8 / max(len(algs), 1)
            offsets = {alg: (i - (len(algs) - 1) / 2) * width for i, alg in enumerate(algs)}

            for alg in algs:
                sub = agg[agg["alg"] == alg].set_index("ranks").reindex(all_ranks).fillna(0.0)
                bottom = np.zeros(len(all_ranks))
                for comp in components:
                    ax.bar(
                        x + offsets[alg], sub[comp], width,
                        bottom=bottom,
                        color=COMPONENT_COLORS[comp],
                        hatch=hatch_for[alg],
                        edgecolor="white",
                        linewidth=0.6,
                    )
                    bottom += sub[comp].to_numpy()

            ax.set_xticks(x)
            ax.set_xticklabels([str(r) for r in all_ranks])
            ax.yaxis.set_major_formatter(FuncFormatter(time_formatter))
            ax.grid(True, axis="y")
            ax.grid(False, axis="x")

            if row == 0:
                ax.set_title(f"{m1}×{n1}×{m2}×{n2}")
            if row == nrows - 1:
                ax.set_xlabel("Ranks")
            if col == 0:
                ax.set_ylabel(f"{op}\nTime")

    component_legend = [
        Patch(facecolor=COMPONENT_COLORS[c], label=COMPONENT_LABELS[c], edgecolor="white")
        for c in components
    ]
    alg_legend = [
        Patch(facecolor="#eeeeee", edgecolor="#444444", hatch=hatch_for[a], label=alg_label(a))
        for a in algs
    ]

    fig.tight_layout()

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.08)

    fig.legend(
        handles=component_legend, loc="lower left", bbox_to_anchor=(0.0, 1.01),
        ncol=len(component_legend), title="Component", title_fontsize=10,
    )
    fig.legend(
        handles=alg_legend, loc="lower right", bbox_to_anchor=(1.0, 1.01),
        ncol=len(alg_legend), title="Algorithm", title_fontsize=10,
    )

    out_path = outdir / f"{out_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Per-rank breakdown ───────────────────────────────────────────────────────

def per_rank_plot(per_rank, outdir):
    per_rank_dir = outdir / "per_rank"
    per_rank_dir.mkdir(parents=True, exist_ok=True)

    combos = per_rank[["m1", "n1", "m2", "n2", "op", "ranks"]].drop_duplicates()
    combos = combos.sort_values(["ranks", "op", "m1", "n1", "m2", "n2"])

    for _, combo in combos.iterrows():
        m1, n1, m2, n2, op, ranks = combo["m1"], combo["n1"], combo["m2"], combo["n2"], combo["op"], combo["ranks"]
        df_c = per_rank[
            (per_rank["m1"] == m1) & (per_rank["n1"] == n1) &
            (per_rank["m2"] == m2) & (per_rank["n2"] == n2) &
            (per_rank["op"] == op) & (per_rank["ranks"] == ranks)
        ]
        algs = sorted(df_c["alg"].unique())
        if not algs:
            continue

        fig, axes = plt.subplots(len(algs), 1, figsize=(11, 3.2 * len(algs)), squeeze=False)

        for i, alg in enumerate(algs):
            ax = axes[i][0]
            df_a = df_c[df_c["alg"] == alg].groupby("rank", as_index=True)[
                ["all_gather", "reduce_scatter", "computation"]
            ].mean().sort_index()

            ranks_idx = df_a.index.to_numpy()
            bottom = np.zeros(len(ranks_idx))
            for comp in ("all_gather", "reduce_scatter", "computation"):
                vals = df_a[comp].fillna(0.0).to_numpy()
                ax.bar(ranks_idx, vals, bottom=bottom, color=COMPONENT_COLORS[comp],
                       edgecolor="white", linewidth=0.3, width=0.9)
                bottom += vals

            ax.set_title(f"{alg_label(alg)}", loc="left", fontsize=11)
            ax.yaxis.set_major_formatter(FuncFormatter(time_formatter))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Rank ID")
            ax.set_ylabel("Time")

        component_legend = [
            Patch(facecolor=COMPONENT_COLORS[c], label=COMPONENT_LABELS[c], edgecolor="white")
            for c in ("all_gather", "reduce_scatter", "computation")
        ]
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.suptitle(f"{op}  •  {m1}×{n1}×{m2}×{n2}  •  {ranks} ranks", fontsize=13, fontweight="bold", y=0.98)
        fig.legend(handles=component_legend, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02))

        out_path = per_rank_dir / f"per_rank_{op}_{m1}x{n1}x{m2}x{n2}_{ranks}ranks.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        print(f"Saved {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV file produced by parse_runs.py --csv")
    ap.add_argument("--outdir", help="Directory to write plots to (default: <csv_dir>/plots)")
    ap.add_argument("--sizes", help="Comma-separated list of MxNxMxN sizes to include (default: all)")
    ap.add_argument("--include-failed", action="store_true", help="Include rows marked failed")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        sys.exit(f"Error: {csv_path} is not a file")

    outdir = Path(args.outdir) if args.outdir else csv_path.resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    df, per_rank = load_csv(csv_path)

    if "failed" in df.columns and not args.include_failed:
        df = df[df["failed"] == 0]

    if df.empty:
        sys.exit("No usable rows in CSV (all failed or empty).")

    sizes = parse_size_filter(args.sizes) if args.sizes else None

    strong_scaling_plot(
        df, outdir, sizes=sizes,
        components=["computation"],
        out_name="strong_scaling_compute_only",
        title="Compute Only",
    )
    strong_scaling_plot(
        df, outdir, sizes=sizes,
        components=["all_gather", "reduce_scatter"],
        out_name="strong_scaling_communication_only",
        title="Communication Only",
    )
    strong_scaling_plot(
        df, outdir, sizes=sizes,
        components=["all_gather", "reduce_scatter", "computation"],
        out_name="strong_scaling_compute_and_communication",
        title="Compute + Communication",
    )

    if per_rank is not None and not per_rank.empty:
        per_rank_plot(per_rank, outdir)
    else:
        print("No per-rank data in CSV; skipping per-rank plots.")


if __name__ == "__main__":
    main()
