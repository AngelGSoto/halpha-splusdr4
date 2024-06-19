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
df_splus = open_csv_conc("Class_main_unique/Halpha_emitter_group*.csv", "simbad")
df_splus_wise = open_csv_conc("Class_wise_main_unique/*.csv", "simbad")

print("Number of objects just with SPLUS colors:", len(df_splus))
print("Number of objects just with SPLUS+WISE colors:", len(df_splus_wise))

# Plotting SPLUS data
def plot_data(df, filename, xlabel, ylabel, loc='upper right'):
    with sns.axes_style("ticks"):
        fig, ax = plt.subplots(figsize=(15, 11), dpi=300)
        plt.xlabel(xlabel, fontsize=32)
        plt.ylabel(ylabel, fontsize=32)
        #plt.title(title, fontsize=36, pad=20)
        plt.tick_params(axis='x', labelsize=30, width=2, length=10)
        plt.tick_params(axis='y', labelsize=30, width=2, length=10)

        # Define colors based on the number of labels
        num_colors = len(df["Label"].unique())
        colors = sns.color_palette("tab10", num_colors)
        edge_colors = sns.color_palette("dark", num_colors)  # Define nice edge colors

        # Iterate over groups and assign colors and markers
        for group, (label, group_data) in enumerate(df.groupby("Label")):
            legend_label = "Noise" if label == -1 else f"Group {label}"
            
            color = colors[group % num_colors]
            edge_color = edge_colors[group % num_colors]
            marker = 'o'  # You can change this if you want different markers for each group
            ax.scatter(group_data["PC1"], group_data["PC2"],
                       c=[color], s=200, marker=marker, edgecolors=[edge_color], linewidth=1.5, label=legend_label)

        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.legend(loc=loc, fontsize=30)

        plt.tight_layout()    
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Plot SPLUS
plot_data(df_splus, "Figs/umap_hdbscan_splus.pdf", "UMAP-1", "UMAP-2")

# Plot SPLUS + WISE
plot_data(df_splus_wise, "Figs/umap_hdbscan_splus_wise.pdf",  "UMAP-1", "UMAP-2", loc='lower left')
