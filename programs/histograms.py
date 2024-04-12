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
plt.savefig("Disk-histogram-r.pdf")

# Calculate the bin width
bin_width_d = (mag_range_d[1] - mag_range_d[0]) / num_bins

# Calculate the number density per magnitude bin
number_density_halpha_d = np.histogram(df_halpha_d["r"], bins=num_bins, range=mag_range_d)[0] / len(df_halpha_d) / bin_width_d / 347.4
number_density_all_objects_d = np.histogram(df_all_objects_d["r"], bins=num_bins, range=mag_range_d)[0] / len(df_all_objects_d) / bin_width_d / 347.4

# Plot number density per magnitude bin
plt.figure(figsize=(10, 6))
plt.plot(number_density_halpha_d, color='blue', label='Hα Excess (Disk)')
plt.plot(number_density_all_objects_d, color='red', label='All Disk Stars')
plt.xlabel("Magnitude Bin")
plt.ylabel("Number Density (mag$^{-1}$ deg$^{-2}$)")
plt.title("Number Density of Objects in r-band Magnitude Bins")
plt.legend()
plt.grid(True)
plt.savefig("Disk-histogram-rpermagdeg.pdf")
