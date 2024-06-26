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

# Load data
df_splus_wise = open_csv_conc("Class_wise_main_unique/*.csv", "simbad")

print("Number of objects just with SPLUS+WISE colors:", len(df_splus_wise))

# Select specific pairs based on domain knowledge or preliminary analysis
specific_pairs = [
    (('u_PStotal', 'J0660_PStotal'), ('W1mag', 'W2mag')),
    (('r_PStotal', 'u_PStotal'), ('W1mag', 'W2mag')),
    (('W2mag', 'z_PStotal'), ('J0378_PStotal', 'J0515_PStotal')),
    (('W1mag', 'z_PStotal'), ('J0378_PStotal', 'J0515_PStotal')),
    (('W2mag', 'i_PStotal'), ('u_PStotal', 'J0660_PStotal')),
    (('i_PStotal', 'z_PStotal'), ('u_PStotal', 'J0660_PStotal')),
    (('W2mag', 'r_PStotal'), ('r_PStotal', 'u_PStotal')),
    (('W1mag', 'i_PStotal'), ('r_PStotal', 'u_PStotal')),
    (('W1mag', 'W2mag'), ('g_PStotal', 'u_PStotal')),
    (('W1mag', 'z_PStotal'), ('g_PStotal', 'u_PStotal')),
    (('i_PStotal', 'z_PStotal'), ('g_PStotal', 'u_PStotal')),
    (('W1mag', 'i_PStotal'), ('i_PStotal', 'u_PStotal')),
    (('i_PStotal', 'z_PStotal'), ('g_PStotal', 'J0378_PStotal')),
    (('W2mag', 'i_PStotal'), ('g_PStotal', 'J0378_PStotal')),
    (('r_PStotal', 'z_PStotal'), ('g_PStotal', 'J0378_PStotal')),
    (('W1mag', 'i_PStotal'), ('g_PStotal', 'J0378_PStotal')),
    (('W1mag', 'z_PStotal'), ('u_PStotal', 'J0515_PStotal')),
    (('i_PStotal', 'z_PStotal'), ('u_PStotal', 'J0515_PStotal')),
    (('W2mag', 'r_PStotal'), ('u_PStotal', 'J0515_PStotal')),
    (('J0378_PStotal', 'J0430_PStotal'), ('u_PStotal', 'J0515_PStotal')),
    (('i_PStotal', 'z_PStotal'), ('u_PStotal', 'J0861_PStotal')),
    (('r_PStotal', 'z_PStotal'), ('u_PStotal', 'J0861_PStotal')),
    (('W1mag', 'i_PStotal'), ('u_PStotal', 'J0861_PStotal')),
    (('r_PStotal', 'u_PStotal'), ('u_PStotal', 'J0430_PStotal')),
    (('i_PStotal', 'z_PStotal'), ('u_PStotal', 'J0430_PStotal')),
    (('W1mag', 'i_PStotal'), ('u_PStotal', 'z_PStotal')),
    (('u_PStotal', 'J0660_PStotal'), ('J0378_PStotal', 'J0430_PStotal')),
    (('J0378_PStotal', 'J0515_PStotal'), ('J0378_PStotal', 'J0430_PStotal')),
    (('W2mag', 'z_PStotal'), ('J0395_PStotal', 'J0515_PStotal')),
    (('i_PStotal', 'z_PStotal'), ('J0395_PStotal', 'J0515_PStotal'))
]

# Create a figure with subplots
fig, axes = plt.subplots(6, 5, figsize=(20, 20), dpi=300)  # 5 rows, 5 columns
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

# Define colors based on the number of labels
num_colors = len(df_splus_wise["Label"].unique())
colors = sns.color_palette("tab10", num_colors)
edge_colors = sns.color_palette("dark", num_colors)  # Define nice edge colors

# Plot each pair of color-color diagrams
for i, ((x1, y1), (x2, y2)) in enumerate(specific_pairs):
    ax = axes[i]

    # Iterate over groups and assign colors and markers
    for group, (label, group_data) in enumerate(df_splus_wise.groupby("Label")):
        legend_label = "Noise" if label == -1 else f"Group {label}"
        color = colors[group % num_colors]
        edge_color = edge_colors[group % num_colors]
        marker = 'o'  # You can change this if you want different markers for each group

        ax.scatter(group_data[x1] - group_data[y1], group_data[x2] - group_data[y2],
                   c=[color], s=50, marker=marker, edgecolors=[edge_color], linewidth=0.5, label=legend_label, alpha=0.7)

    # Customize axis labels with the Wise filter renaming
    if x1 == 'W1mag':
        xlabel = 'W1'
    elif x1 == 'W2mag':
        xlabel = 'W2'
    else:
        xlabel = x1.replace('_PStotal', '')

    if y1 == 'W1mag':
        ylabel = 'W1'
    elif y1 == 'W2mag':
        ylabel = 'W2'
    else:
        ylabel = y1.replace('_PStotal', '')

    ax.set_xlabel(f"{xlabel} - {ylabel}", fontsize=22)

    if x2 == 'W1mag':
        xlabel = 'W1'
    elif x2 == 'W2mag':
        xlabel = 'W2'
    else:
        xlabel = x2.replace('_PStotal', '')

    if y2 == 'W1mag':
        ylabel = 'W1'
    elif y2 == 'W2mag':
        ylabel = 'W2'
    else:
        ylabel = y2.replace('_PStotal', '')

    ax.set_ylabel(f"{xlabel} - {ylabel}", fontsize=22)

    ax.tick_params(axis='both', labelsize=22)
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Set aspect ratio to be square without changing subplot dimensions
    ax.set_aspect('auto')

    # Add legend after all subplots, centered horizontally at the bottom
    if i == 0:  # Add legend only to the first subplot
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=22, ncol=7)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("Figs/color_color_diagrams_multiple-extended.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

