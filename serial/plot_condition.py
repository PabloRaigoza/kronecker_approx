import matplotlib.pyplot as plt
import numpy as np

with open('cond_benchmark.txt', 'r') as f:
    data_lines = f.readlines()

# Dictionary to hold category-wise data
categories = {
    10: [],
    20: [],
    30: [],
    40: [],
    50: [],
    60: [],
}

# Parse each line
for line in data_lines:
    fields = line.strip().split(',')

    category = int(fields[0])  # e.g., 10, 20, ..., 60
    cond_number = float(fields[4])  # condition number
    rel_error = float(fields[-1])   # last column = relative error

    if category in categories:
        categories[category].append((cond_number, rel_error))
    else:
        print(f"Unknown category {category} in line: {line.strip()}")
import matplotlib.pyplot as plt
import numpy as np

# Colors like the example chart
colors = ["#4A90E2", "#9B9B9B", "#D0021B", "#7ED321", "#F5A623", "#9013FE"]

# Get all unique condition numbers across all categories, sorted
all_conds = sorted(set(cond for data in categories.values() for cond, _ in data))
cond_to_index = {cond: i for i, cond in enumerate(all_conds)}

# X positions
x = np.arange(len(all_conds))
bar_width = 0.12
num_categories = len(categories)
labels = sorted(categories.keys())

fig, ax = plt.subplots(figsize=(12, 6))

for i, category in enumerate(labels):
    data = categories[category]
    x_vals, y_vals = [], []

    for cond, err in data:
        idx = cond_to_index[cond]
        x_offset = idx + (i - num_categories / 2) * bar_width + bar_width / 2
        x_vals.append(x_offset)
        y_vals.append(err)

    # Bars
    bars = ax.bar(
        x_vals, y_vals, width=bar_width,
        label=f"{category**2}x{category**2}",
        color=colors[i % len(colors)]
    )

# Axis labels and ticks
ax.set_xticks(x)
ax.set_xticklabels([f"{cond:.1f}" for cond in all_conds])
ax.set_xlabel("Condition Number")
ax.set_ylabel("Relative Error")
ax.set_title("Relative Error vs Condition Number by Category", fontweight="bold")
# plot y=sqrt2
ax.axhline(y=np.sqrt(2), color='r', linestyle='--', label='y=√2', linewidth=1)
# ax.set_yscale("log")
# Legend on top, like example
ax.legend()

# Grid lines
ax.grid(axis="y", linestyle="--", alpha=0.6)

# plt.tight_layout()
# plt.show()
plt.savefig("condition_number_vs_rel_error.png", dpi=300)

