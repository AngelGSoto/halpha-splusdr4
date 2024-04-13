import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

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
    color='#006d2c',  # Change color to a more standard one
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


############################################################################################3
# Create new DataFrame for galactic latitude and longitude
# Convert RA and DEC to galactic latitude and longitude
df_d_ = df_d.rename(columns={"ALPHA": "RA", "DELTA": "DEC"})
coords_d = SkyCoord(ra=df_d_["RA"].values * u.degree, dec=df_d_["DEC"].values * u.degree, frame='icrs')
coords_h = SkyCoord(ra=df_h["RA"].values * u.degree, dec=df_h["DEC"].values * u.degree, frame='icrs')
galactic_coords_d = coords_d.galactic
galactic_coords_h = coords_h.galactic

# Set seaborn style
sns.set_style("whitegrid")

# Define colors for disk and main
color_disk = '#66c2a5'  # Greenish color
color_main = '#fc8d62'  # Orangish color

# Define border colors for disk and main
border_color_disk = '#006d2c'  # Dark green
border_color_main = '#8c2d04'  # Dark orange

# Plot latitude versus longitude
plt.figure(figsize=(10, 6))

# Scatter plot for disk
plt.scatter(galactic_coords_d.l.deg, galactic_coords_d.b.deg, color=color_disk, edgecolor=border_color_disk, linewidth=1, label='Disk', s=30, alpha=0.5)

# Scatter plot for main
plt.scatter(galactic_coords_h.l.deg, galactic_coords_h.b.deg, color=color_main, edgecolor=border_color_main, linewidth=1, label='Main', s=30, alpha=0.5)

plt.xlabel("Galactic Longitude (l)", fontsize=20)
plt.ylabel("Galactic Latitude (b)", fontsize=20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)

plt.legend(loc='upper center', fontsize=25)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Figs/Halpha_galactic_coordinates.pdf")


