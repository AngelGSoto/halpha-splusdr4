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
plt.xlabel("UMAP-1", fontsize=30)
plt.ylabel("UMAP-2", fontsize=30)
plt.tick_params(axis='x', labelsize=30, width=2, length=10)  # Adjusting width and length of tick marks
plt.tick_params(axis='y', labelsize=30, width=2, length=10)  # Adjusting width and length of tick marks

# Create a scatter plot with different marker styles and colors
markers = ['o', 's', 'D', '^']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Set custom colors
for group, marker, color in zip(range(4), markers, colors):
    border_color = np.array(plt.cm.colors.to_rgba(color))  # Convert color to RGBA array
    border_color *= 0.6  # Darken the color by reducing its intensity
    ax.scatter(df_splus["PC1"][df_splus["Label"] == group], 
               df_splus["PC2"][df_splus["Label"] == group], 
               c=color, s=200, marker=marker, edgecolors=border_color, linewidth=1.5, label=f"Group {group}")

# Add minor tick locators without showing the minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

# Customize legend with larger font size
ax.legend(loc='upper left', fontsize=25)

plt.tight_layout()    
plt.savefig("Figs/umap_hdbscam_splus_disk.pdf")

#######################################################
#SPLUS + WISE                           ···············
#######################################################

fig, ax = plt.subplots(figsize=(15, 11))
plt.xlabel("UMAP-1", fontsize=30)
plt.ylabel("UMAP-2", fontsize=30)
plt.tick_params(axis='x', labelsize=30, width=2, length=10)  # Adjusting width and length of tick marks
plt.tick_params(axis='y', labelsize=30, width=2, length=10)  # Adjusting width and length of tick marks

# Create a scatter plot with different marker styles and colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#ff9896']

markers = ['o', 's', 'D', '^', '*', 'P']

for group, marker, color in zip(range(6), markers, colors):
    border_color = np.array(plt.cm.colors.to_rgba(color))  # Convert color to RGBA array
    border_color *= 0.6  # Darken the color by reducing its intensity
    ax.scatter(df_splus_wise["PC1"][df_splus_wise["Label"] == group], 
               df_splus_wise["PC2"][df_splus_wise["Label"] == group], 
               c=color, s=200, marker=marker, edgecolors=border_color, linewidth=1.5, label=f"Group {group}")

# Add minor tick locators without showing the minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

# Customize legend with larger font size
ax.legend(loc='lower center', fontsize=25)

plt.tight_layout()    
plt.savefig("Figs/umap_hdbscam_splus_wise_disk.pdf")
