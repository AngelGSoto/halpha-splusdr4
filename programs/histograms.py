import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
import seaborn as sns
sns.set_theme(style="ticks")

# Open the CSV files
# Define the pattern to match CSV files
pattern = 'Disk_Lomeli/*.csv'

# Use glob to find all CSV files in the current directory
csv_files = glob.glob(pattern)

# Create an empty list to store DataFrames
dfs = []

# Loop through each CSV file and read it into a DataFrame
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
df_all_objects_d = pd.concat(dfs, ignore_index=True)

# The Halpha emitters
df_halpha_d = pd.read_csv("Ha-emitters-disk/Halpha-disk_splus_Mine_PerField_total.csv")

# Define the number of bins and the range of r-band magnitudes
num_bins = 50
mag_range_d = (min(df_halpha_d["r"].min(), df_all_objects_d["r"].min()), max(df_halpha_d["r"].max(), df_all_objects_d["r"].max()))

# Plot normalized histograms of r-band magnitudes for Hα emitters and all objects
plt.figure(figsize=(10, 6))
plt.hist(df_halpha_d["r"], bins=num_bins, range=mag_range_d, alpha=0.5, color='blue', label='Hα Excess (Disk)', density=True)
plt.hist(df_all_objects_d["r"], bins=num_bins, range=mag_range_d, alpha=0.5, color='red', label='All Disk Stars', density=True)
plt.xlabel("r-band Magnitude")
plt.ylabel("Normalized Density")
plt.legend()
plt.grid(True)
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
plt.plot(fraction_halpha, color='blue', label='Fraction of Hα Emitters (Disk)')
plt.xlabel("Magnitude Bin")
plt.ylabel("Fraction of Hα Emitters (mag$^{-1}$ deg$^{-2}$)")
plt.grid(True)
plt.savefig("Figs/Fraction-Halpha-Disk.pdf")

