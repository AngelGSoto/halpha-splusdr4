import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
import seaborn as sns
sns.set_theme(style="ticks")

# Define colors for disk and main
color_disk = '#66c2a5'  # Greenish color
color_main = '#fc8d62'  # Orangish color
border_color_disk = '#006d2c'  # Dark green
border_color_main = '#8c2d04'  # Dark orange

# Load data
df_d = pd.read_csv("Ha-emitters-disk-iteractive/Halpha-disk_splus_Mine_PerField_total-unique.csv")
df_h = pd.read_csv("Ha-emitters/Halpha_Mine_PerField_total-unique.csv")

# Calculate colors
ri_d = df_d["r"] - df_d["i"]
rj660_d = df_d["r"] - df_d["F660"]
ri_h = df_h["r_PStotal"] - df_h["i_PStotal"]
rj660_h = df_h["r_PStotal"] - df_h["J0660_PStotal"]

# Plotting Halpha-colorDistribution
fig, ax = plt.subplots(figsize=(15, 11))
plt.xlabel(r"$r - i$", fontsize=35)
plt.ylabel(r"$r - J0660$", fontsize=35)
plt.tick_params(axis='both', labelsize=30, width=3, length=12)

# Scatter plot for disk
scatter_disk = ax.scatter(
    ri_d, rj660_d,
    color=color_disk,
    alpha=0.8,
    s=400,
    marker='o',
    edgecolors=border_color_disk,
    linewidths=1,
    zorder=2, label="GDS"
)

# Scatter plot for main
scatter_main = ax.scatter(
    ri_h, rj660_h,
    color=color_main,
    alpha=0.4,
    s=300,
    marker='s',
    edgecolors=border_color_main,
    linewidths=1,
    zorder=2, label="MS"
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

# Create new DataFrame for galactic latitude and longitude
# Convert RA and DEC to galactic latitude and longitude
df_d_ = df_d.rename(columns={"ALPHA": "RA", "DELTA": "DEC"})
coords_d = SkyCoord(ra=df_d_["RA"].values * u.degree, dec=df_d_["DEC"].values * u.degree, frame='icrs')
coords_h = SkyCoord(ra=df_h["RA"].values * u.degree, dec=df_h["DEC"].values * u.degree, frame='icrs')
galactic_coords_d = coords_d.galactic
galactic_coords_h = coords_h.galactic

# Plotting Halpha_galactic_coordinates
plt.figure(figsize=(10, 6))

# Scatter plot for disk
plt.scatter(galactic_coords_d.l.deg, galactic_coords_d.b.deg, color=color_disk, edgecolor=border_color_disk, linewidth=1, label='GDS', s=30, alpha=0.5)

# Scatter plot for main
plt.scatter(galactic_coords_h.l.deg, galactic_coords_h.b.deg, color=color_main, edgecolor=border_color_main, linewidth=1, label='MS', s=30, alpha=0.5)

plt.xlabel("Galactic Longitude (l)", fontsize=25)
plt.ylabel("Galactic Latitude (b)", fontsize=25)
plt.tick_params(axis='both', labelsize=20)
plt.legend(loc='upper center', bbox_to_anchor=(0.4, 1.0), fontsize=25)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Figs/Halpha_galactic_coordinates.pdf")

# Plotting Halpha_galactic_coordinates_disk
plt.figure(figsize=(10, 6))

# KDE plot for disk
sns.kdeplot(x=galactic_coords_d.l.deg, y=galactic_coords_d.b.deg, cmap="Greens", fill=True, bw_adjust=0.5)

# Scatter plot for disk
plt.scatter(galactic_coords_d.l.deg, galactic_coords_d.b.deg, color=color_disk, edgecolor=border_color_disk, linewidth=1, label='GDS', s=30, alpha=0.5)

plt.xlabel("Galactic Longitude (l)", fontsize=25)
plt.ylabel("Galactic Latitude (b)", fontsize=25)
plt.tick_params(axis='both', labelsize=20)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Figs/Halpha_galactic_coordinates_disk.pdf")
