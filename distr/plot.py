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

Golub-Kahan (op "bidiag") rows produce, in addition:
    gkb_breakdown_percent.png          share of GKB time: Ax, ATx, Allreduce, vector ops
    gkb_compute_only.png               GKB computation (dgemv + local vector ops)
    gkb_communication_only.png         GKB communication (All Gather, Reduce Scatter, Allreduce)
    gkb_compute_and_communication.png  both, stacked
    scaling_lines_matvec.png           Ax+ATx computation / communication / both vs ranks,
    scaling_lines_gkb.png              and the same for GKB; rows group matrices of similar Ã size,
                                       one line per matrix, colored by scheme
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
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
    numeric = [c for c in df.columns if c not in ("file", "op", "alg")]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

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


# ── Golub-Kahan bidiagonalization ────────────────────────────────────────────

# Same hues as the matvec plots for the shared components; the GK-only
# pieces get their own.
GKB_COLORS = {
    "all_gather": COMPONENT_COLORS["all_gather"],
    "reduce_scatter": COMPONENT_COLORS["reduce_scatter"],
    "allreduce": "#8172B3",
    "computation": COMPONENT_COLORS["computation"],
    "vector_ops": "#937860",
    "barrier": "#BBBBBB",
}
GKB_LABELS = {
    "all_gather": "All Gather (Ax+ATx)",
    "reduce_scatter": "Reduce Scatter (Ax+ATx)",
    "allreduce": "Allreduce (norms)",
    "computation": "Computation (dgemv)",
    "vector_ops": "Vector ops (local)",
    "barrier": "Barrier wait (imbalance)",
}
# Breakdown by algorithm stage: matvec kernels in greens, the rest of GK apart
BREAKDOWN_COLORS = {
    "ax": "#2E7D4F",
    "atx": "#8CCB9B",
    "allreduce": GKB_COLORS["allreduce"],
    "vector_ops": GKB_COLORS["vector_ops"],
    "barrier": GKB_COLORS["barrier"],
}
BREAKDOWN_LABELS = {
    "ax": "Ax kernel",
    "atx": "ATx kernel",
    "allreduce": "Allreduce (norms)",
    "vector_ops": "Vector ops (local)",
    "barrier": "Barrier wait (imbalance)",
}

SIZE_COLS = ["m1", "n1", "m2", "n2"]


def size_label(size):
    return "×".join(str(int(v)) for v in size)


def gkb_frame(df):
    """
    One row per bidiag experiment with additive components (seconds):
      all_gather, reduce_scatter, allreduce, computation, vector_ops,
      ax, atx, total, approx.

    Runs that print "Bidiag Mean Breakdown" give per-rank means, which add
    up exactly. Older runs only have per-component maxima (taken on
    different ranks, so they overshoot when stacked); for those the
    kernel / non-kernel split follows the measured per-rank-mean kernel
    fraction, and each side is divided in proportion to its maxima.
    approx=True marks those rows.
    """
    g = df[df["op"].astype(str).str.startswith("bidiag")].copy()
    if g.empty:
        return g

    def col(name):
        return g[name] if name in g.columns else pd.Series(np.nan, index=g.index)

    exact = col("mean_total").notna()

    ag_x = col("mean_ax_all_gather") + col("mean_atx_all_gather")
    rs_x = col("mean_ax_reduce_scatter") + col("mean_atx_reduce_scatter")
    comp_x = col("mean_ax_computation") + col("mean_atx_computation")
    ax_x = col("mean_ax_all_gather") + col("mean_ax_computation") + col("mean_ax_reduce_scatter")
    atx_x = col("mean_atx_all_gather") + col("mean_atx_computation") + col("mean_atx_reduce_scatter")

    total_a = col("bd_total")
    kernel_a = col("kernel_frac_mean") * total_a
    ax_max = col("bd_ax_all_gather") + col("bd_ax_computation") + col("bd_ax_reduce_scatter")
    atx_max = col("bd_atx_all_gather") + col("bd_atx_computation") + col("bd_atx_reduce_scatter")
    k_scale = kernel_a / (ax_max + atx_max).replace(0, np.nan)
    other_a = total_a - kernel_a
    o_scale = other_a / (col("bd_reductions") + col("bd_vector_ops")).replace(0, np.nan)

    g["all_gather"] = np.where(exact, ag_x, (col("bd_ax_all_gather") + col("bd_atx_all_gather")) * k_scale)
    g["reduce_scatter"] = np.where(exact, rs_x, (col("bd_ax_reduce_scatter") + col("bd_atx_reduce_scatter")) * k_scale)
    g["computation"] = np.where(exact, comp_x, (col("bd_ax_computation") + col("bd_atx_computation")) * k_scale)
    g["ax"] = np.where(exact, ax_x, ax_max * k_scale)
    g["atx"] = np.where(exact, atx_x, atx_max * k_scale)
    g["allreduce"] = np.where(exact, col("mean_reductions"), col("bd_reductions") * o_scale)
    g["vector_ops"] = np.where(exact, col("mean_vector_ops"), col("bd_vector_ops") * o_scale)
    # Time waiting in GKB sync barriers; 0 for runs without them (there the
    # wait is folded into the collective that follows)
    g["barrier"] = np.where(exact, col("mean_barrier").fillna(0.0), 0.0)
    g["total"] = np.where(exact, col("mean_total"), total_a)
    g["approx"] = ~exact
    comps = ["all_gather", "reduce_scatter", "computation", "ax", "atx", "allreduce", "vector_ops", "barrier"]
    g[comps] = g[comps].fillna(0.0)
    return g


def _wrapped_axes(n, ncols=5, panel_w=4.6, panel_h=3.9):
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(panel_w * ncols, 7.5), panel_h * nrows), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n], ncols


def _sorted_sizes(frame, sizes=None):
    all_sizes = sorted(frame[SIZE_COLS].drop_duplicates().itertuples(index=False, name=None),
                       key=lambda s: (s[0] * s[1] * s[2] * s[3], s))
    if sizes:
        all_sizes = [s for s in all_sizes if s in sizes]
    return all_sizes


def gkb_grid_plot(g, outdir, components, colors, labels, out_name, title, percent=False, sizes=None):
    """One panel per matrix: x = ranks, grouped bars per algorithm (hatched),
    each bar stacked by component. percent=True normalizes each bar to its
    GK total (components that are not plotted still count in the total)."""
    all_sizes = _sorted_sizes(g, sizes)
    if not all_sizes:
        print(f"No GKB data; skipping {out_name}.")
        return

    algs = sorted(g["alg"].unique())
    hatch_for = {a: HATCHES[i % len(HATCHES)] for i, a in enumerate(algs)}
    fig, axes, ncols = _wrapped_axes(len(all_sizes))

    for i, (size, ax) in enumerate(zip(all_sizes, axes)):
        df_size = g[(g[SIZE_COLS] == size).all(axis=1)]
        agg = df_size.groupby(["alg", "ranks"], as_index=False)[list(components) + ["total"]].mean()
        all_ranks = sorted(agg["ranks"].unique())
        x = np.arange(len(all_ranks))
        width = 0.8 / max(len(algs), 1)

        for j, alg in enumerate(algs):
            sub = agg[agg["alg"] == alg].set_index("ranks").reindex(all_ranks).fillna(0.0)
            denom = sub["total"].replace(0, np.nan).to_numpy() if percent else 1.0
            bottom = np.zeros(len(all_ranks))
            for comp in components:
                vals = np.nan_to_num(sub[comp].to_numpy() / denom)
                ax.bar(x + (j - (len(algs) - 1) / 2) * width, vals, width, bottom=bottom,
                       color=colors[comp], hatch=hatch_for[alg], edgecolor="white", linewidth=0.6)
                bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in all_ranks])
        if percent:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(time_formatter))
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.set_title(size_label(size))
        if i >= len(all_sizes) - ncols:
            ax.set_xlabel("Ranks")
        if i % ncols == 0:
            ax.set_ylabel("Share of GKB time" if percent else "GKB time")

    component_legend = [Patch(facecolor=colors[c], label=labels[c], edgecolor="white") for c in components]
    alg_legend = [Patch(facecolor="#eeeeee", edgecolor="#444444", hatch=hatch_for[a], label=alg_label(a))
                  for a in algs]
    fig.tight_layout()
    k = g["steps"].dropna()
    steps = f", k={int(k.max())} steps" if not k.empty else ""
    note = ""
    if g["approx"].any():
        note = "\n(approx. breakdown: these runs predate per-rank mean timings; split follows the mean kernel fraction)"
    fig.suptitle(f"{title}{steps}{note}", fontsize=14, fontweight="bold", y=1.10 if note else 1.08)
    fig.legend(handles=component_legend, loc="lower left", bbox_to_anchor=(0.0, 1.01),
               ncol=len(component_legend), title="Component", title_fontsize=10)
    fig.legend(handles=alg_legend, loc="lower right", bbox_to_anchor=(1.0, 1.01),
               ncol=len(alg_legend), title="Algorithm", title_fontsize=10)

    out_path = outdir / f"{out_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved {out_path}")


# One color per partitioning scheme (first three categorical slots, which
# stay distinguishable for color-vision deficiencies)
ALG_COLORS = {"wbp": "#2a78d6", "rrp": "#eb6834", "bcp": "#1baf7a"}
FALLBACK_ALG_COLORS = ["#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def size_gb(size):
    return size[0] * size[1] * size[2] * size[3] * 8 / 1e9


def group_sizes_by_gb(sizes, ratio=2.0):
    """Group matrices of similar Ã size: walk sizes in increasing GB and start
    a new group when a matrix is more than `ratio`x the group's smallest."""
    groups = []
    for size in sorted(sizes, key=lambda s: (size_gb(s), s)):
        if groups and size_gb(size) <= ratio * size_gb(groups[-1][0]):
            groups[-1].append(size)
        else:
            groups.append([size])
    return groups


def _gb_text(gb):
    return f"{gb / 1e3:.2g} TB" if gb >= 1e3 else f"{gb:.0f} GB"


def _group_title(group):
    lo, hi = size_gb(group[0]), size_gb(group[-1])
    rng = _gb_text(lo) if _gb_text(lo) == _gb_text(hi) else f"{_gb_text(lo)} – {_gb_text(hi)}"
    return f"Ã {rng}"


LINE_COLUMNS = [
    ("Computation", True, False),
    ("Communication", False, True),
    ("Computation + Communication", True, True),
]


def _label_line_ends(ax, ends, fontsize=7.5):
    """Label each matrix once, in a column just right of the axes.
    ends: list of (label, x_end, y_end). Labels are spread apart vertically
    so they don't overlap; a thin leader line connects each label to its
    line end. Works on linear or log y axes."""
    if not ends:
        return
    log = ax.get_yscale() == "log"
    fwd = np.log10 if log else (lambda v: v)
    inv = (lambda v: 10 ** v) if log else (lambda v: v)
    lo, hi = fwd(np.array(ax.get_ylim()))
    # Minimum gap between labels as a fraction of the axis height
    height_pt = ax.get_window_extent().height * 72 / ax.figure.dpi
    gap = (hi - lo) * min(0.2, 1.35 * fontsize / max(height_pt, 1.0))
    ends = sorted(ends, key=lambda e: e[2])
    ys = [fwd(e[2]) for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    overflow = ys[-1] - (hi - gap / 2)
    if overflow > 0:  # shift down, then re-push from the bottom limit
        ys = [y - overflow for y in ys]
        ys[0] = max(ys[0], lo + gap / 2)
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for (label, x_end, y_end), y in zip(ends, ys):
        ax.annotate(label, xy=(x_end, y_end), xycoords="data", xytext=(1.03, inv(y)), textcoords=trans,
                    fontsize=fontsize, color="#444444", va="center", ha="left", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="#b0b0b0", linewidth=0.5, shrinkA=0, shrinkB=2))


def lines_grid_plot(data, outdir, out_name, title, sizes=None, note=""):
    """Rows = groups of similar-size matrices; columns = computation /
    communication / both (linear y, independent per panel). One line per
    (matrix, scheme), colored by scheme; matrices labeled at the line ends.
    data: SIZE_COLS + alg, ranks, comp, comm."""
    all_sizes = _sorted_sizes(data, sizes)
    if not all_sizes:
        return
    groups = group_sizes_by_gb(all_sizes)
    algs = sorted(data["alg"].unique())
    extra = iter(FALLBACK_ALG_COLORS)
    color_for = {a: ALG_COLORS.get(a) or next(extra) for a in algs}
    all_ranks = sorted(data["ranks"].unique())

    fig, axes = plt.subplots(len(groups), len(LINE_COLUMNS),
                             figsize=(6.8 * len(LINE_COLUMNS), 3.6 * len(groups)), squeeze=False)
    panel_ends = {}
    for r, group in enumerate(groups):
        for c, (metric, use_comp, use_comm) in enumerate(LINE_COLUMNS):
            ax = axes[r][c]
            vals = data.assign(value=(data["comp"] if use_comp else 0.0) + (data["comm"] if use_comm else 0.0))
            last = {}
            for alg in algs:
                for size in group:
                    sub = vals[(vals["alg"] == alg) & (vals[SIZE_COLS] == size).all(axis=1)]
                    sub = sub.groupby("ranks", as_index=False)["value"].mean().sort_values("ranks")
                    if sub.empty:
                        continue
                    ax.plot(sub["ranks"], sub["value"], marker="o", markersize=3.5, linewidth=1.6,
                            alpha=0.85, color=color_for[alg])
                    last.setdefault(size, []).append((sub["ranks"].iloc[-1], sub["value"].iloc[-1]))
            # One label per matrix at the mean of its schemes' line ends
            panel_ends[(r, c)] = [
                (size_label(size), max(p[0] for p in pts), float(np.mean([p[1] for p in pts])))
                for size, pts in last.items()
            ]
            ax.set_xscale("log", base=2)
            ax.set_xticks(all_ranks)
            ax.set_xticklabels([str(x) for x in all_ranks])
            ax.minorticks_off()
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(FuncFormatter(time_formatter))
            if r == 0:
                ax.set_title(metric)
            if r == len(groups) - 1:
                ax.set_xlabel("Ranks")
            if c == 0:
                ax.set_ylabel(f"{_group_title(group)}\nTime")

    alg_legend = [Line2D([0], [0], color=color_for[a], linewidth=2, marker="o", markersize=4, label=alg_label(a))
                  for a in algs]
    # Title block height in figure fraction (one more line when there's a note)
    n_title_lines = 2 + note.count("\n")
    title_frac = (0.27 * n_title_lines + 0.45) / fig.get_figheight()
    fig.tight_layout(rect=[0, 0, 0.93, 1 - title_frac])
    fig.subplots_adjust(wspace=0.55)
    for (r, c), ends in panel_ends.items():
        _label_line_ends(axes[r][c], ends)
    fig.suptitle(f"{title}\n(rows: matrices grouped by Ã size; one line per matrix and scheme){note}",
                 fontsize=14, fontweight="bold", y=1 - 0.08 / fig.get_figheight(), va="top")
    fig.legend(handles=alg_legend, loc="upper right", bbox_to_anchor=(1.0, 0.995), ncol=len(algs),
               title="Algorithm", title_fontsize=10)

    out_path = outdir / f"{out_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved {out_path}")


def scaling_lines_plots(df, g, outdir, sizes=None):
    """Ax+ATx (one of each) and GKB, as computation / communication / both."""
    keys = SIZE_COLS + ["alg", "ranks"]
    parts = []
    for op in ("Ax", "ATx"):
        sub = df[df["op"] == op].copy()
        if sub.empty:
            continue
        sub["comp"] = sub["computation"].fillna(0.0)
        sub["comm"] = sub["all_gather"].fillna(0.0) + sub["reduce_scatter"].fillna(0.0)
        parts.append(sub.groupby(keys, as_index=False)[["comp", "comm"]].mean())
    if len(parts) == 2:
        # Only (matrix, scheme, ranks) that have both an Ax and an ATx result
        mv = parts[0].merge(parts[1], on=keys, suffixes=("_ax", "_atx"))
        mv["comp"] = mv["comp_ax"] + mv["comp_atx"]
        mv["comm"] = mv["comm_ax"] + mv["comm_atx"]
        lines_grid_plot(mv, outdir, "scaling_lines_matvec", "Strong scaling: Ax + ATx (one of each)", sizes)

    if g is not None and not g.empty:
        gk = g.copy()
        gk["comp"] = gk["computation"] + gk["vector_ops"]
        gk["comm"] = gk["all_gather"] + gk["reduce_scatter"] + gk["allreduce"]
        k = g["steps"].dropna()
        k_txt = f", k={int(k.max())} steps" if not k.empty else ""
        note = "\n(approx. split: runs predate per-rank mean timings)" if g["approx"].any() else ""
        lines_grid_plot(gk, outdir, "scaling_lines_gkb", f"Strong scaling: Golub-Kahan bidiagonalization{k_txt}",
                        sizes, note=note)


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

    gkb = gkb_frame(df)
    df_mv = df[df["op"].isin(["Ax", "ATx"])]

    if not df_mv.empty:
        matvec_plots(df_mv, outdir, sizes)

    if not gkb.empty:
        gkb_grid_plot(gkb, outdir, ["ax", "atx", "allreduce", "vector_ops", "barrier"], BREAKDOWN_COLORS, BREAKDOWN_LABELS,
                      "gkb_breakdown_percent", "GKB Runtime Breakdown", percent=True, sizes=sizes)
        gkb_grid_plot(gkb, outdir, ["computation", "vector_ops"], GKB_COLORS, GKB_LABELS,
                      "gkb_compute_only", "GKB Compute Only", sizes=sizes)
        gkb_grid_plot(gkb, outdir, ["all_gather", "reduce_scatter", "allreduce"], GKB_COLORS, GKB_LABELS,
                      "gkb_communication_only", "GKB Communication Only", sizes=sizes)
        gkb_grid_plot(gkb, outdir, ["all_gather", "reduce_scatter", "allreduce", "computation", "vector_ops", "barrier"],
                      GKB_COLORS, GKB_LABELS, "gkb_compute_and_communication", "GKB Compute + Communication",
                      sizes=sizes)

    scaling_lines_plots(df_mv, gkb, outdir, sizes)

    if per_rank is not None and not per_rank.empty:
        per_rank_plot(per_rank, outdir)
    else:
        print("No per-rank data in CSV; skipping per-rank plots.")


def matvec_plots(df, outdir, sizes):
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


if __name__ == "__main__":
    main()
