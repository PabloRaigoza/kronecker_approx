import re
import pandas as pd
from pathlib import Path

def time_format(x, pos):
    if x >= 1:
        return f"{x:.0f}s"
    elif x >= 0.001:
        return f"{x*1e3:.0f}ms"
    else:
        return f"{x*1e6:.0f}µs"

alg_map = {
    "wbp": "ax_alg1",
    "rrp": "ax_alg2",
    "bcp": "ax_alg3",
}

def parse_experiment_log(path) -> pd.DataFrame:
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()

    header_pattern = re.compile(
        r"Experiment:\s+"
        r"(\d+)x(\d+)x(\d+)x(\d+)\s+"  # m1 n1 m2 n2
        r"(\d+)\s+"                   # ranks
        r"(\S+)\s+"                   # op
        r"ranks\s+"
        r"(\S+)"                      # algorithm
    )

    # Mean patterns (all optional per line)
    all_gather_pattern = re.compile(r"Mean All Gather:\s*([\d.]+)")
    computation_pattern = re.compile(r"Mean Computation:\s*([\d.]+)")
    reduce_scatter_pattern = re.compile(r"Mean Reduce Scatter:\s*([\d.]+)")

    rows = []
    current_meta = None

    for line in lines:
        # Check for experiment header
        header_match = header_pattern.search(line)
        if header_match:
            m1, n1, m2, n2, ranks, op, algorithm = header_match.groups()
            
            current_meta = {
                "Algorithm": alg_map.get(algorithm, algorithm),
                "Op": op,
                "Ranks": int(ranks),
                "m1": int(m1),
                "n1": int(n1),
                "m2": int(m2),
                "n2": int(n2),
            }
            continue

        if current_meta is None:
            continue

        # Parse mean line(s)
        all_gather_match = all_gather_pattern.search(line)
        computation_match = computation_pattern.search(line)
        reduce_scatter_match = reduce_scatter_pattern.search(line)

        # Only create a row if at least one metric is found
        if any([all_gather_match, computation_match, reduce_scatter_match]):
            row = {
                **current_meta,
                "All_Gather_Time": float(all_gather_match.group(1)) if all_gather_match else 0.0,
                "Computation_Time": float(computation_match.group(1)) if computation_match else 0.0,
                "Reduce_Scatter_Time": float(reduce_scatter_match.group(1)) if reduce_scatter_match else 0.0,
            }
            rows.append(row)

            # Reset so we don’t accidentally reuse metadata if format breaks
            current_meta = None

    return rows

# data = parse_experiment_log("node_1_48803154.out")
# data.extend(parse_experiment_log("node_2_48803229.out"))
# data.extend(parse_experiment_log("node_4_48803691.out"))
# data.extend(parse_experiment_log("node_8_48803692.out"))
# data = parse_experiment_log("runs/run2/node_1_48806370.out")
# data.extend(parse_experiment_log("runs/run2/node_2_48806372.out"))
# data.extend(parse_experiment_log("runs/run2/node_4_48806374.out"))
# data.extend(parse_experiment_log("runs/run2/node_8_48806377.out"))
# data.extend(parse_experiment_log("runs/run2/node_16_48808959.out"))
# data.extend(parse_experiment_log("runs/run2/node_32_48809096.out"))

# data = parse_experiment_log("node_1_51627998.out")
# data.extend(parse_experiment_log("node_2_51628001.out"))
# data.extend(parse_experiment_log("node_4_51628672.out"))

data = parse_experiment_log("node_1_51628876.out")
data.extend(parse_experiment_log("node_2_51628878.out"))
data.extend(parse_experiment_log("node_4_51628879.out"))
data.extend(parse_experiment_log("node_8_51638642.out"))
data.extend(parse_experiment_log("node_16_51629130.out"))


data = pd.DataFrame(data)
# print(data[:129])

agg = (
    data
    .groupby(
        ["Algorithm", "Op", "m1", "n1", "m2", "n2", "Ranks"],
        as_index=False
    )
    .agg(
        AllGather_mean=("All_Gather_Time", "mean"),
        Compute_mean=("Computation_Time", "mean"),
        ReduceScatter_mean=("Reduce_Scatter_Time", "mean"),
    )
)


def strong_scaling_bar_plot(df, sizes):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    operations = ["Ax", "ATx"]
    ncols = len(sizes)

    colors = {
        "All_Gather_Time": "#ffbb78",
        "Reduce_Scatter_Time": "#ff7f0e",
        "Computation_Time": "#2ca02c"
    }

    fig, axes = plt.subplots(
        2, ncols,
        figsize=(5*ncols, 10),
        sharey=False,
        sharex=False
    )

    # If only one column, axes shape is (2,), fix indexing
    if ncols == 1:
        axes = np.array(axes).reshape(2,1)

    for col, PROBLEM_SIZE in enumerate(sizes):

        m1, n1, m2, n2 = PROBLEM_SIZE

        df_size = df[
            (df["m1"] == m1) &
            (df["n1"] == n1) &
            (df["m2"] == m2) &
            (df["n2"] == n2)
        ]

        for row, OPERATION in enumerate(operations):

            ax = axes[row, col]

            df_op = df_size[df_size["Op"] == OPERATION]
            if df_op.empty:
                continue

            agg = (
                df_op
                .groupby(["Algorithm", "Ranks"], as_index=False)
                .agg(
                    All_Gather_Time=("All_Gather_Time", "mean"),
                    Reduce_Scatter_Time=("Reduce_Scatter_Time", "mean"),
                    Computation_Time=("Computation_Time", "mean"),
                )
            )

            algorithms = sorted(agg["Algorithm"].unique())
            all_ranks = sorted(agg["Ranks"].unique())

            x = np.arange(len(all_ranks))
            width = 0.25

            offsets = {
                algo: (i - (len(algorithms) - 1) / 2) * width
                for i, algo in enumerate(algorithms)
            }

            for algo in algorithms:
                subset = agg[agg["Algorithm"] == algo]
                subset = subset.set_index("Ranks").reindex(all_ranks, fill_value=0.0)

                all_gather = subset["All_Gather_Time"]
                reduce_scatter = subset["Reduce_Scatter_Time"]
                computation = subset["Computation_Time"]

                bottom_rs = all_gather
                bottom_comp = all_gather + reduce_scatter

                hatch_map = {
                    "ax_alg1": "//",
                    "ax_alg2": "\\\\",
                    "ax_alg3": None,
                }
                hatch = hatch_map.get(algo, None)

                ax.bar(x + offsets[algo], all_gather, width,
                       color=colors["All_Gather_Time"], hatch=hatch)

                ax.bar(x + offsets[algo], reduce_scatter, width,
                       bottom=bottom_rs,
                       color=colors["Reduce_Scatter_Time"], hatch=hatch)

                # ax.bar(x + offsets[algo], computation, width,
                #        bottom=bottom_comp,
                #        color=colors["Computation_Time"], hatch=hatch)

            # ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)

            # Column titles (problem size only on top row)
            if row == 0:
                ax.set_title(
                    f"{m1}x{n1}x{m2}x{n2}",
                    fontsize=16
                )
                ax.set_xticks(x)
                ax.set_xticklabels([r // 1 for r in all_ranks], fontsize=12)
                ax.set_xlabel("Cores")               
                

            # Row labels on first column only
            if col == 0:
                ax.set_ylabel(f"{OPERATION}\nTime", fontsize=16)

            # X ticks (bottom row only)
            if row == 1:
                ax.set_xticks(x)
                ax.set_xticklabels([r // 1 for r in all_ranks], fontsize=12)
                ax.set_xlabel("Cores")

    # Fix y-axis formatting
    for ax in axes.flatten():
        ax.yaxis.set_major_formatter(plt.FuncFormatter(time_format))

    # -----------------------
    # Global Legends
    # -----------------------
    stack_legend = [
        Patch(facecolor=colors["All_Gather_Time"], label="All Gather"),
        Patch(facecolor=colors["Reduce_Scatter_Time"], label="Reduce Scatter"),
        # Patch(facecolor=colors["Computation_Time"], label="Computation"),
    ]

    # algo_legend = [
    #     Patch(facecolor="white", edgecolor="black", hatch="//", label=""),
    #     Patch(facecolor="white", edgecolor="black", hatch="\\\\", label="BCP"),
    #     Patch(facecolor="white", edgecolor="black", label="RRP"),
    # ]
    # find the key in alg_map that corresponds to "ax_alg1" and make the label "WBP"
    algo_legend = [
        Patch(facecolor="white", edgecolor="black", hatch="//", label="ax_alg1"),
        Patch(facecolor="white", edgecolor="black", hatch="\\\\", label="ax_alg2"),
        Patch(facecolor="white", edgecolor="black", label="ax_alg3"),
    ]

    fig.legend(
        handles=stack_legend,
        loc="upper left",
        ncol=3,
        title="Components",
        fontsize=12
    )

    fig.legend(
        handles=algo_legend,
        loc="upper right",
        title="Algorithm",
        fontsize=12
    )

    fig.suptitle("", fontsize=22)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig("strong_scaling_grid.png", dpi=300)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def rank_time_bar_plot(df, PROBLEM_SIZE, OPERATION, ALGORITHM, TOTAL_RANKS):
    m1, n1, m2, n2 = PROBLEM_SIZE

    # Filter fixed problem size, operation, algorithm, and total ranks
    df = df[
        (df["m1"] == m1) & (df["n1"] == n1) &
        (df["m2"] == m2) & (df["n2"] == n2) &
        (df["Op"] == OPERATION) &
        (df["Algorithm"] == ALGORITHM) &
        (df["Ranks"] == TOTAL_RANKS)
    ]

    if df.empty:
        print("No data for this configuration")
        return

    # Ensure each rank is unique
    df = df.groupby("Rank", as_index=True).agg(
        All_Gather_Time=("All_Gather_Time", "mean"),
        Reduce_Scatter_Time=("Reduce_Scatter_Time", "mean"),
        Computation_Time=("Computation_Time", "mean"),
    )

    all_ranks = sorted(df.index)
    all_gather = df["All_Gather_Time"]
    reduce_scatter = df["Reduce_Scatter_Time"]
    computation = df["Computation_Time"]

    bottom_rs = all_gather
    bottom_comp = all_gather + reduce_scatter

    colors = {
        "All_Gather_Time": "#ffbb78",
        "Reduce_Scatter_Time": "#ff7f0e",
        "Computation_Time": "#2ca02c"
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(all_ranks, all_gather, color=colors["All_Gather_Time"], label="All Gather")
    ax.bar(all_ranks, reduce_scatter, bottom=bottom_rs, color=colors["Reduce_Scatter_Time"], label="Reduce Scatter")
    ax.bar(all_ranks, computation, bottom=bottom_comp, color=colors["Computation_Time"], label="Computation")

    ax.set_yscale("log")
    ax.set_xlabel("Rank ID")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{ALGORITHM} — {OPERATION} ({m1}x{n1}x{m2}x{n2}), {TOTAL_RANKS} ranks")

    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"rank_time_{ALGORITHM}_{OPERATION}_{TOTAL_RANKS}ranks.png", dpi=300)
    plt.close()

def rank_time_bar_plot_total(df, PROBLEM_SIZES, OPERATIONS, ALGORITHMS, TOTAL_RANKS):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {
        "All_Gather_Time": "#ffbb78",
        "Reduce_Scatter_Time": "#ff7f0e",
        "Computation_Time": "#2ca02c"
    }

    # Prepare all combinations
    combinations = []
    for ps in PROBLEM_SIZES:
        for alg in ALGORITHMS:
            for op in OPERATIONS:
                combinations.append((ps, op, alg))

    n_subplots = len(combinations)
    n_cols = 4
    n_rows = (n_subplots + n_cols - 1) // n_cols  # ceil division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), sharey=False)
    axes = axes.flatten()

    # for ax in axes:
    #     ax.set_ylim(1e-4, 10 * 1.2)  # adjust lower bound as appropriate

    for idx, (ps, op, alg) in enumerate(combinations):
        ax = axes[idx]

        m1, n1, m2, n2 = ps
        df_sub = df[
            (df["m1"] == m1) & (df["n1"] == n1) &
            (df["m2"] == m2) & (df["n2"] == n2) &
            (df["Op"] == op) &
            (df["Algorithm"] == alg) &
            (df["Ranks"] == TOTAL_RANKS)
        ]

        if df_sub.empty:
            ax.set_title(f"{op} | {alg} | {m1}x{n1}x{m2}x{n2}\nNo data")
            ax.axis("off")
            continue

        # Keep rank order as in the log
        ranks = df_sub["Rank"].values
        all_gather = df_sub["All_Gather_Time"].values
        reduce_scatter = df_sub["Reduce_Scatter_Time"].values
        computation = df_sub["Computation_Time"].values

        eps = 1e-12
        all_gather = np.maximum(all_gather, eps)
        reduce_scatter = np.maximum(reduce_scatter, eps)
        computation = np.maximum(computation, eps)


        bottom_rs = all_gather
        bottom_comp = all_gather + reduce_scatter

        hatch = "//" if alg == "ax_alg1" else None

        ax.bar(ranks, all_gather, color=colors["All_Gather_Time"], hatch=hatch, label="All Gather")
        ax.bar(ranks, reduce_scatter, bottom=bottom_rs, color=colors["Reduce_Scatter_Time"], hatch=hatch, label="Reduce Scatter")
        ax.bar(ranks, computation, bottom=bottom_comp, color=colors["Computation_Time"], hatch=hatch, label="Computation")

        ax.set_title(f"{op} | {alg} | {m1}x{n1}x{m2}x{n2}", fontsize=10)
        # ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.3)
        ax.set_xlabel("Rank ID", fontsize=8)

    # Hide unused subplots
    for ax in axes[n_subplots:]:
        ax.axis("off")

    for ax in axes:
        ax.relim()
        ax.autoscale_view()


    # Fix y-axis formatting
    for ax in axes[:n_subplots]:
        ax.set_ylabel("Time (s)", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(time_format))

    # Global legend
    stack_legend = [
        Patch(facecolor=colors["All_Gather_Time"], label="All Gather"),
        Patch(facecolor=colors["Reduce_Scatter_Time"], label="Reduce Scatter"),
        Patch(facecolor=colors["Computation_Time"], label="Computation"),
    ]
    algo_legend = [
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Alg 1"),
        Patch(facecolor="white", edgecolor="black", label="Alg 2"),
    ]

    fig.legend(handles=stack_legend + algo_legend, loc="lower center", ncol=5, fontsize=10, title="Legend")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle(f"Rank-wise Total Time (Stacked) Across Experiments {TOTAL_RANKS}", fontsize=16)
    plt.savefig(f"rank_time_bar_plot_all_raw_order_{TOTAL_RANKS}.png", dpi=300)
    plt.close()


# strong_scaling_plot(agg, 200, 200, 1000, 1000)
# rank_time_bar_plot(data, PROBLEM_SIZE=(1000, 1000, 200, 200), OPERATION="Ax", ALGORITHM="ax_alg1", TOTAL_RANKS=128)

strong_scaling_bar_plot(data, [(100, 100, 100, 100), (200, 200, 1000, 1000), (1000, 1000, 200, 200), (475, 475, 475, 475)])
# rank_time_bar_plot_total(
#     data,
#     PROBLEM_SIZES=[(100, 100, 100, 100), (200, 200, 1000, 1000), (1000, 1000, 200, 200), (475, 475, 475, 475)],
#     OPERATIONS=["Ax", "ATx"],
#     ALGORITHMS=["ax_alg1", "ax_alg2"],
#     TOTAL_RANKS=512
# )