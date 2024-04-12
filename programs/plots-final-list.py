import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
import seaborn as sns
sns.set_theme(style="ticks")

# Load data
df_d = pd.read_csv("Ha-emitters-disk/Halpha-disk_splus_Mine_PerField_total.csv")
df_h = pd.read_csv("Ha-emitters/Halpha_Mine_PerField_total.csv")

# Calculate colors
ri_d = df_d["r"] - df_d["i"]
rj660_d = df_d["r"] - df_d["F660"]

ri_h = df_h["r_PStotal"] - df_h["i_PStotal"]
rj660_h = df_h["r_PStotal"] - df_h["J0660_PStotal"]

# Plotting
fig, ax = plt.subplots(figsize=(15, 11))
plt.xlabel(r"$r - i$", fontsize=35)
plt.ylabel(r"$r - J0660$", fontsize=35)
plt.tick_params(axis='x', labelsize=30, width=3, length=12)
plt.tick_params(axis='y', labelsize=30, width=3, length=12)

# Scatter plot for disk
scatter_disk = ax.scatter(
    ri_d, rj660_d,
    color='skyblue',  # Change color to a more standard one
    alpha=0.8,  # Add transparency
    s=600,  # Decrease marker size
    marker='o',  # Use circles for disk
    edgecolors="k",
    linewidths=1,
    zorder=2, label="Disk"
)

# Scatter plot for main
scatter_main = ax.scatter(
    ri_h, rj660_h,
    color='salmon',  # Change color to a more standard one
    alpha=0.4,  # Add transparency
    s=600,  # Increase marker size
    marker='s',  # Use squares for main
    edgecolors="k",
    linewidths=1,
    zorder=2, label="Main"
)

# Add minor tick locators without showing the minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

# Customize legend with larger font size
ax.legend(loc='upper left', fontsize=35)

plt.tight_layout()    
plt.savefig("Figs/Halpha-colorDistribution.pdf")

