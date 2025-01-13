import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
import seaborn as sns

def open_csv_conc(pattern, exclude_pattern):
    csv_files = glob.glob(pattern)
    csv_files = [file for file in csv_files if exclude_pattern not in file]
    dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def find_intersection(m1, b1, m2, b2):
    x_intersect = (b2 - b1) / (m1 - m2)
    y_intersect = m1 * x_intersect + b1
    return x_intersect, y_intersect

def add_custom_cut_lines(ax, m1, b1, m2, b2, color):
    x_intersect, y_intersect = find_intersection(m1, b1, m2, b2)
    x_range1 = np.linspace(ax.get_xlim()[0], x_intersect, 200)
    y_range1 = m1 * x_range1 + b1
    x_range2 = np.linspace(x_intersect, ax.get_xlim()[1], 200)
    y_range2 = m2 * x_range2 + b2
    ax.plot(x_range1, y_range1, color=color, linestyle='-', linewidth=2)
    ax.plot(x_range2, y_range2, color=color, linestyle='-', linewidth=2)
    ax.scatter([x_intersect], [y_intersect], color=color, s=50, zorder=5)

# Load data
df_splus_wise = open_csv_conc("Class_wise_main_unique/*.csv", "simbad")

print("Number of objects just with SPLUS+WISE colors:", len(df_splus_wise))

# Select specific pairs based on domain knowledge or preliminary analysis
specific_pairs = [
    (('W2mag', 'r_PStotal'), ('J0378_PStotal', 'J0430_PStotal')),
    (('J0395_PStotal', 'J0430_PStotal'), ('W2mag', 'i_PStotal')),
    (('i_PStotal', 'z_PStotal'), ('W2mag', 'z_PStotal')),
    (('u_PStotal', 'J0430_PStotal'), ('W1mag', 'W2mag')),
    (('g_PStotal', 'u_PStotal'), ('W1mag', 'i_PStotal')),
    (('W2mag', 'g_PStotal'), ('g_PStotal', 'J0378_PStotal')),
    (('J0430_PStotal', 'J0515_PStotal'), ('i_PStotal', 'J0861_PStotal')),
    (('W1mag', 'z_PStotal'), ('z_PStotal', 'J0660_PStotal')),
    (('W1mag', 'r_PStotal'), ('r_PStotal', 'J0378_PStotal'))
]

# Define custom cut lines for each class and each plot
custom_cut_lines = {
    0: {
        0: [(0.5, 1.0), (-1.0, 3.0)], # Example values for class 0, plot 0
        1: [(0.3, 1.2), (-0.7, 2.8)], # Example values for class 0, plot 1
        # Add more plots here
    },
    1: {
        0: [(0.2, 0.5), (-0.8, 2.5)], # Example values for class 1, plot 0
        1: [(0.4, 0.9), (-1.2, 3.2)], # Example values for class 1, plot 1
        # Add more plots here
    },
    # Add more classes here
}

# Create a figure with subplots
fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=300)
axes = axes.flatten()

# Define colors based on the number of labels
num_colors = len(df_splus_wise["Label"].unique())
colors = sns.color_palette("tab10", num_colors)
edge_colors = sns.color_palette("dark", num_colors)

# Plot each pair of color-color diagrams
for i, ((x1, y1), (x2, y2)) in enumerate(specific_pairs):
    ax = axes[i]

    all_x = []
    all_y = []
    points_per_class = []

    for group, (label, group_data) in enumerate(df_splus_wise.groupby("Label")):
        legend_label = "Noise" if label == -1 else f"Group {label}"
        color = colors[group % num_colors]
        edge_color = edge_colors[group % num_colors]
        marker = 'o'

        x_values = group_data[x1] - group_data[y1]
        y_values = group_data[x2] - group_data[y2]
        points = np.column_stack((x_values, y_values))
        points_per_class.append((points, color, label))
        all_x.extend(x_values)
        all_y.extend(y_values)

        ax.scatter(x_values, y_values,
                   c=[color], s=90, marker=marker, edgecolors=[edge_color], linewidth=0.5, label=legend_label, alpha=0.7)

    xlabel = x1.replace('_PStotal', '').replace('W1mag', 'W1').replace('W2mag', 'W2')
    ylabel = y1.replace('_PStotal', '').replace('W1mag', 'W1').replace('W2mag', 'W2')
    ax.set_xlabel(f"{xlabel} - {ylabel}", fontsize=22)

    xlabel = x2.replace('_PStotal', '').replace('W1mag', 'W1').replace('W2mag', 'W2')
    ylabel = y2.replace('_PStotal', '').replace('W1mag', 'W1').replace('W2mag', 'W2')
    ax.set_ylabel(f"{xlabel} - {ylabel}", fontsize=22, labelpad=10)

    ax.tick_params(axis='both', labelsize=22)
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_aspect('auto')

    # Calculate dynamic axis limits with a margin
    margin = 0.2
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add custom cut lines to the plots for each class
    for points, color, label in points_per_class:
        if label in custom_cut_lines and i in custom_cut_lines[label]:
            m1, b1 = custom_cut_lines[label][i][0]
            m2, b2 = custom_cut_lines[label][i][1]
            add_custom_cut_lines(ax, m1, b1, m2, b2, color)

    if i == 0:
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=22, ncol=7)

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.tight_layout()
plt.savefig("Figs/color_color_diagrams_multiple_balanced_ColorCut.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()
