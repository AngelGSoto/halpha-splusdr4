import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.coordinates import SkyCoord
import astropy.units as u

# Set seaborn style
sns.set_theme(style="ticks")

# Open the CSV files
pattern = 'Disk_Lomeli/*.csv'
csv_files = glob.glob(pattern)
dfs = []

# Loop through each CSV file and read it into a DataFrame
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
df_all_objects_d = pd.concat(dfs, ignore_index=True)

# The Halpha emitters
df_halpha_d = pd.read_csv("Ha-emitters-disk/Halpha-disk_splus_Mine_PerField_total-unique.csv")

# Define the number of bins and the range of r-band magnitudes
num_bins = 50
mag_range_d = (min(df_halpha_d["r"].min(), df_all_objects_d["r"].min()), max(df_halpha_d["r"].max(), df_all_objects_d["r"].max()))

# Plot side-by-side histograms of r-band magnitudes for Hα emitters and all objects
plt.figure(figsize=(10, 6))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.hist(df_halpha_d["r"], bins=num_bins, range=mag_range_d, alpha=0.7, color='deepskyblue', label='Hα Excess (Disk)', density=True, zorder =3)
plt.hist(df_all_objects_d["r"], bins=num_bins, range=mag_range_d, alpha=0.7, color='salmon', label='All Disk Stars', density=True, zorder =2)
plt.xlabel("r-band Magnitude", fontsize=20)
plt.ylabel("Normalized Density", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("Figs/Disk-histogram-r.pdf")

# Calculate the bin width
bin_width_d = (mag_range_d[1] - mag_range_d[0]) / num_bins

# Calculate the number density per magnitude bin
number_density_halpha_d = np.histogram(df_halpha_d["r"], bins=num_bins, range=mag_range_d)[0] / bin_width_d / 347.4
number_density_all_objects_d = np.histogram(df_all_objects_d["r"], bins=num_bins, range=mag_range_d)[0] / bin_width_d / 347.4

# Calculate the fraction of Hα emitters in each magnitude bin
fraction_halpha = number_density_halpha_d / (number_density_halpha_d + number_density_all_objects_d)

# Plot fraction of Hα emitters in each r-band magnitude bin
plt.figure(figsize=(10, 6))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.plot(fraction_halpha, color='navy', label='Fraction of Hα Emitters (Disk)')
plt.xlabel("Magnitude Bin", fontsize=20)
plt.ylabel("Fraction of Hα Emitters (mag$^{-1}$ deg$^{-2}$)", fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("Figs/Fraction-Halpha-Disk.pdf")

# Plot number density of Hα emitters and all objects in each r-band magnitude bin as a bar plot
plt.figure(figsize=(10, 6))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)

# Define the bin edges
bin_edges = np.arange(len(number_density_halpha_d) + 1) - 0.5

# Plot the histograms with step style
plt.hist(np.arange(len(number_density_halpha_d)), bins=bin_edges, weights=number_density_halpha_d, color='deepskyblue', label='Hα Excess (Disk)', alpha=1, zorder =3,
         histtype='step', linewidth=3, density=True)
plt.hist(np.arange(len(number_density_all_objects_d)), bins=bin_edges, weights=number_density_all_objects_d, color='salmon', label='All Disk Stars', alpha=1,  zorder =2,
         histtype='step', linewidth=3, density=True)

plt.xlabel("r-band Magnitude Bin", fontsize=20)  # Add clarification
plt.ylabel("Number Density (mag$^{-1}$ deg$^{-2}$)", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True, axis='y')  # Add grid lines only on the y-axis
plt.xticks(np.arange(0, len(number_density_halpha_d), 5), rotation=45)  # Set x-ticks to align with the bins and rotate them
plt.tight_layout()  # Improve spacing between subplots
plt.savefig("Figs/Number-Density-Disk.pdf")

# Plot cumulative distribution function (CDF) of r-band magnitudes
plt.figure(figsize=(10, 6))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.hist(df_halpha_d["r"], bins=num_bins, range=mag_range_d, alpha=1, color='deepskyblue', label='Hα Excess (Disk)', cumulative=True, density=True,  zorder =3)
plt.hist(df_all_objects_d["r"], bins=num_bins, range=mag_range_d, alpha=1, color='salmon', label='All Disk Stars', cumulative=True, density=True,  zorder =2)
plt.xlabel("r-band Magnitude")
plt.ylabel("Cumulative Density")
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("Figs/Disk-cdf-r.pdf")

# Plot Kernel Density Estimation (KDE) of r-band magnitudes
plt.figure(figsize=(10, 6))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
sns.kdeplot(df_halpha_d["r"], color='navy', label='Hα Excess (Disk)', linestyle='-', linewidth=2,  zorder =3)
sns.kdeplot(df_all_objects_d["r"], color='salmon', label='All Disk Stars', linestyle='--', linewidth=2,  zorder =2)
plt.xlabel("r-band Magnitude", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("Figs/Disk-kde-r.pdf")

##############################################################################

# Convert RA and DEC to galactic latitude and longitude
df_d_ = df_all_objects_d.rename(columns={"ALPHA": "RA", "DELTA": "DEC"})
coords_d = SkyCoord(ra=df_d_["RA"].values * u.degree, dec=df_d_["DEC"].values * u.degree, frame='icrs')
galactic_coords_d = coords_d.galactic

# Calculate Galactic Longitude for Hα emitters
df_d_h = df_halpha_d.rename(columns={"ALPHA": "RA", "DELTA": "DEC"})
coords_h = SkyCoord(ra=df_d_h["RA"].values * u.degree, dec=df_d_h["DEC"].values * u.degree, frame='icrs')
galactic_coords_h = coords_h.galactic
df_halpha_d["GAL_LONG"] = galactic_coords_h.l.deg  # Add Galactic Longitude to the DataFrame

# Define the number of bins and the range of Galactic Longitude
num_bins = 50
lon_range = (min(galactic_coords_d.l.deg), max(galactic_coords_d.l.deg))

# Plot side-by-side histograms of Galactic Longitude for Hα emitters and all objects
plt.figure(figsize=(10, 6))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.hist(df_halpha_d["GAL_LONG"], bins=num_bins, range=lon_range,  color='deepskyblue', label='Hα Excess (Disk)', alpha=1, zorder=3,
         histtype='step', linewidth=3, density=True)
plt.hist(galactic_coords_d.l.deg, bins=num_bins, range=lon_range,  color='salmon', label='All Disk Stars', alpha=1, zorder=2,
         histtype='step', linewidth=3, density=True)
plt.xlabel("Galactic Longitude (l)", fontsize=20)
plt.ylabel("Normalized Density", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("Figs/Disk-histogram-galactic-longitude.pdf")

# Calculate the bin width
bin_width = (lon_range[1] - lon_range[0]) / num_bins

# Calculate the number density per Galactic Longitude bin
number_density_all_objects_d = np.histogram(galactic_coords_d.l.deg, bins=num_bins, range=lon_range)[0] / bin_width / 347.4
number_density_halpha_d = np.histogram(df_halpha_d["GAL_LONG"], bins=num_bins, range=lon_range)[0] / bin_width / 347.4

# Calculate the fraction of Hα emitters in each Galactic Longitude bin
fraction_halpha = number_density_halpha_d / (number_density_halpha_d + number_density_all_objects_d)

# Plot fraction of Hα emitters in each Galactic Longitude bin
plt.figure(figsize=(10, 6))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.plot(fraction_halpha, color='navy', label='Fraction of Hα Emitters (Disk)')
plt.xlabel("Galactic Longitude (l)", fontsize=20)
plt.ylabel("Fraction of Hα Emitters (mag$^{-1}$ deg$^{-2}$)", fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("Figs/Fraction-Halpha-Galactic-Longitude.pdf")

# Plot number density of Hα emitters and all objects in each Galactic Longitude bin as a bar plot
plt.figure(figsize=(10, 6))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)

# Define the bin edges
bin_edges = np.arange(len(number_density_halpha_d) + 1) - 0.5

# Plot the histograms with step style
plt.hist(np.arange(len(number_density_halpha_d)), bins=bin_edges, weights=number_density_halpha_d, color='deepskyblue', label='Hα Excess (Disk)', alpha=1, zorder=3,
         histtype='step', linewidth=3, density=True)
plt.hist(np.arange(len(number_density_all_objects_d)), bins=bin_edges, weights=number_density_all_objects_d, color='salmon', label='All Disk Stars', alpha=1, zorder=2,
         histtype='step', linewidth=3, density=True)

plt.xlabel("Galactic Longitude (l) bin", fontsize=20)  # Add clarification
plt.ylabel("Number Density (mag$^{-1}$ deg$^{-2}$)", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True, axis='y')  # Add grid lines only on the y-axis
plt.xticks(np.arange(0, len(number_density_halpha_d), 5), rotation=45)  # Set x-ticks to align with the bins and rotate them
plt.tight_layout()  # Improve spacing between subplots
plt.savefig("Figs/Number-Density-Galactic-Longitude.pdf")


# Plot cumulative distribution function (CDF) of Galactic Longitude
plt.figure(figsize=(10, 6))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.hist(df_halpha_d["GAL_LONG"], bins=num_bins, range=lon_range, alpha=1, color='deepskyblue', label='Hα Excess (Disk)', cumulative=True, density=True, zorder=3)
plt.hist(galactic_coords_d.l.deg, bins=num_bins, range=lon_range, alpha=1, color='salmon', label='All Disk Stars', cumulative=True, density=True, zorder=2)
plt.xlabel("Galactic Longitude", fontsize=20)
plt.ylabel("Cumulative Density", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("Figs/Disk-cdf-galactic-longitude.pdf")

# Plot Kernel Density Estimation (KDE) of Galactic Longitude
plt.figure(figsize=(10, 6))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
sns.kdeplot(df_halpha_d["GAL_LONG"], color='navy', label='Hα Excess (Disk)', linestyle='-', linewidth=2, zorder=3)
sns.kdeplot(galactic_coords_d.l.deg, color='salmon', label='All Disk Stars', linestyle='--', linewidth=2, zorder=2)
plt.xlabel("Absolute Galactic Longitude", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("Figs/Disk-kde-galactic-longitude.pdf")
