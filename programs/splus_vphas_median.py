from astropy.io import fits
import numpy as np
from astropy.table import Table
import pandas as pd
import glob
from astropy.table import vstack
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm, median_abs_deviation

# Set seaborn style
sns.set_style("whitegrid")

# Load your DataFrame from CSV
df = pd.read_csv("Ha-emitters-disk/Halpha-disk_splus_Mine_PerField_total-unique_vphas.csv")
print("Number of sources with VPHAS match:", len(df))

# Compute ri colors for SPLUS and VPHAS
ri_splus = df["r"] - df["i"]
ri_vphas = df["rmag"] - df["imag"]

# Compute rj660 colors for SPLUS and VPHAS
rj660_splus = df["r"] - df["F660"]
rj660_vphas = df["rmag"] - df["Hamag"]

# Compute differences between r-i and rj660 colors of SPLUS and VPHAS
ri_diff = ri_splus - ri_vphas
rj660_diff = rj660_splus - rj660_vphas

# Remove non-finite values
ri_diff = ri_diff[np.isfinite(ri_diff)]
rj660_diff = rj660_diff[np.isfinite(rj660_diff)]

# Plot histogram of differences for ri color
plt.figure(figsize=(8, 6))
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)

plt.hist(ri_diff, bins=25, color='#4CAF50', edgecolor='black', alpha=0.7, density=True)  # Green color for bars

# Compute median and MAD
median_ri = np.median(ri_diff)
mad_ri = median_abs_deviation(ri_diff)

# Plot the fitted Gaussian curve
plt.axvline(median_ri, color='blue', linestyle='-', linewidth=1.5)  # Blue solid line for median
plt.axvline(median_ri + mad_ri, color='purple', linestyle='-.', linewidth=1.5)  # Purple dash-dot line for median + MAD
plt.axvline(median_ri - mad_ri, color='purple', linestyle='-.', linewidth=1.5)  # Purple dash-dot line for median - MAD

# Add median and MAD to the plot
plt.text(0.25, 0.95, f"Median: {median_ri:.2f}\nMAD: {mad_ri:.2f}", verticalalignment='top', horizontalalignment='right', transform=plt.gca().transAxes, fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
print("Median (r - i)(S-PLUS - VPHAS+):",  median_ri, "MAD (r - i)(S-PLUS - VPHAS+):", mad_ri)
plt.xlabel('Difference in $r - i$ color (S-PLUS - VPHAS+)', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.legend(['Median', 'Median ± MAD'], loc='upper right', fontsize=15)

plt.savefig("Figs/comparison_SPLUS_VPHAS_ri_with_gaussian_fit_median.pdf")

##################################################################################################################

# Plot histogram of differences for rj660 color
plt.figure(figsize=(8, 6))
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
plt.hist(rj660_diff, bins=25, color='#4CAF50', edgecolor='black', alpha=0.7, density=True)  # Green color for bars

# Compute median and MAD
median_rj660 = np.median(rj660_diff)
mad_rj660 = median_abs_deviation(rj660_diff)

# Plot the fitted Gaussian curve
plt.axvline(median_rj660, color='blue', linestyle='-', linewidth=1.5)  # Blue solid line for median
plt.axvline(median_rj660 + mad_rj660, color='purple', linestyle='-.', linewidth=1.5)  # Purple dash-dot line for median + MAD
plt.axvline(median_rj660 - mad_rj660, color='purple', linestyle='-.', linewidth=1.5)  # Purple dash-dot line for median - MAD

# Add median and MAD to the plot
plt.text(0.25, 0.95, f"Median: {median_rj660:.2f}\nMAD: {mad_rj660:.2f}", verticalalignment='top', horizontalalignment='right', transform=plt.gca().transAxes, fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
print("Median (r - Halpa)(S-PLUS - VPHAS+):",  median_rj660, "MAD (r - Halpa)(S-PLUS - VPHAS+):", mad_rj660)
plt.xlabel(r'Difference in $r - H\alpha$ color (S-PLUS - VPHAS+)', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.legend(['Median', 'Median ± MAD'], loc='upper right', fontsize=15)

plt.savefig("Figs/comparison_SPLUS_VPHAS_rj660_with_gaussian_fit_median.pdf")
