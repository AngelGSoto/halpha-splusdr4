"""
This script is for plotting the results of UMAP and HDBSCAM.
"""
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
import seaborn as sns; sns.set()
sns.set_theme(style="ticks")

def open_csv_conc(pattern):
    # pattern = '../Ha-emitters/*PerField_wise.csv'
    # Use glob to find all CSV files in the current directory
    csv_files = glob.glob(pattern)
    # Create an empty list to store DataFrames
    dfs = []

    # Loop through each CSV file and read it into a DataFrame
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

# Results using only the 66 SPLUS colors
df_splus = open_csv_conc("Class_disk/*.csv")
print("Number of objects just with SPLUS colors:", len(df_splus))

# Results using only the 66 SPLUS colors plus WISE
df_splus_wise = open_csv_conc("Class_disk_wise/*.csv")
print("Number of objects just with SPLUS+WISE colors:", len(df_splus_wise))

# Plotting
fig, ax = plt.subplots(figsize=(15, 11))
plt.xlabel("UMAP-1", fontsize=32)
plt.ylabel("UMAP-2", fontsize=32)
plt.tick_params(axis='x', labelsize=32, width=2, length=10)  # Adjusting width and length of tick marks
plt.tick_params(axis='y', labelsize=32, width=2, length=10)  # Adjusting width and length of tick marks

# Define colors based on the number of CSV files found
num_colors = len(df_splus["Label"].unique())
colors = sns.color_palette("tab10", num_colors)
edge_colors = sns.color_palette("dark", num_colors)  # Define nice edge colors

# Iterate over groups and assign colors and markers
for group, (label, group_data) in enumerate(df_splus.groupby("Label")):
    if label == -1:
        legend_label = "Noise"
    else:
        legend_label = f"Group {label}"
        
    color = colors[group % num_colors]
    edge_color = edge_colors[group % num_colors]
    marker = 'o'  # You can change this if you want different markers for each group
    ax.scatter(group_data["PC1"], group_data["PC2"],
               c=[color], s=200, marker=marker, edgecolors=[edge_color], linewidth=1.5, label=legend_label)

# Add minor tick locators without showing the minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

# Customize legend with larger font size
ax.legend(loc='lower left', fontsize=30)

plt.tight_layout()    
plt.savefig("Figs/umap_hdbscam_splus_disk.pdf")

#######################################################
#SPLUS + WISE                           ···············
#######################################################

fig, ax = plt.subplots(figsize=(15, 11))
plt.xlabel("UMAP-1", fontsize=32)
plt.ylabel("UMAP-2", fontsize=32)
plt.tick_params(axis='x', labelsize=32, width=2, length=10)  # Adjusting width and length of tick marks
plt.tick_params(axis='y', labelsize=32, width=2, length=10)  # Adjusting width and length of tick marks

# Define colors based on the number of CSV files found
num_colors_wise = len(df_splus_wise["Label"].unique())
colors_wise = sns.color_palette("tab10", num_colors_wise)
edge_colors_wise = sns.color_palette("dark", num_colors_wise)  # Define nice edge colors

# Iterate over groups and assign colors and markers
for group, (label, group_data) in enumerate(df_splus_wise.groupby("Label")):
    if label == -1:
        legend_label = "Noise"
    else:
        legend_label = f"Group {label}"
        
    color_wise = colors_wise[group % num_colors_wise]
    edge_color_wise = edge_colors_wise[group % num_colors_wise]
    marker_wise = 'o'  # You can change this if you want different markers for each group
    ax.scatter(group_data["PC1"], group_data["PC2"],
               c=[color_wise], s=200, marker=marker_wise, edgecolors=[edge_color_wise], linewidth=1.5, label=legend_label)

# Add minor tick locators without showing the minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

# Customize legend with larger font size
ax.legend(loc='upper right', fontsize=30)

plt.tight_layout()    
plt.savefig("Figs/umap_hdbscam_splus_wise_disk.pdf")
