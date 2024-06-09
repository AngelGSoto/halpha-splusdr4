from astropy.io import fits
import numpy as np
from astropy.table import Table
import pandas as pd
import glob
from astropy.table import vstack
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Set seaborn style
sns.set_style("whitegrid")

# Load your DataFrame from CSV
df = pd.read_csv("Ha-emitters-disk/Halpha-disk_splus_Mine_PerField_total-unique_vphas.csv")
print("Number of sorces with VPHAS match:", len(df))

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

# Fit Gaussian to the data
mu, std = norm.fit(ri_diff)
x = np.linspace(min(ri_diff), max(ri_diff), 100)
p = norm.pdf(x, mu, std)

# Plot the fitted Gaussian curve
plt.plot(x, p, 'r--', linewidth=2)  # Red dashed line for Gaussian fit

# Add mean and standard deviation to the plot
plt.text(0.25, 0.95, f"Mean: {mu:.2f}\nStd Dev: {std:.2f}", verticalalignment='top', horizontalalignment='right', transform=plt.gca().transAxes, fontsize=15, bbox=dict(facecolor='white', alpha=0.5))

# Add vertical lines for mean and mean +/- std
plt.axvline(mu, color='blue', linestyle='-', linewidth=1.5)  # Blue solid line for mean
plt.axvline(mu + std, color='purple', linestyle='-.', linewidth=1.5)  # Purple dash-dot line for mean + std
plt.axvline(mu - std, color='purple', linestyle='-.', linewidth=1.5)  # Purple dash-dot line for mean - std

plt.xlabel('Difference in $r - i$ color (S-PLUS - VPHAS+)', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.legend(['Gaussian Fit', 'Mean', 'Mean ± Std Dev'], loc='upper right', fontsize=15)

plt.savefig("Figs/comparison_SPLUS_VPHAS_ri_with_gaussian_fit.pdf")

##################################################################################################################

# Plot histogram of differences for rj660 color
plt.figure(figsize=(8, 6))
plt.tick_params(axis='x', labelsize=15) 
plt.tick_params(axis='y', labelsize=15)
plt.hist(rj660_diff, bins=25, color='#4CAF50', edgecolor='black', alpha=0.7, density=True)  # Blue color for bars

# Fit Gaussian to the data
mu_, std_ = norm.fit(rj660_diff)
x_ = np.linspace(min(rj660_diff), max(rj660_diff), 100)
p_ = norm.pdf(x_, mu_, std_)

# Plot the fitted Gaussian curve
plt.plot(x_, p_, 'r--', linewidth=2)  # Red dashed line for Gaussian fit

# Add mean and standard deviation to the plot
plt.text(0.25, 0.95, f"Mean: {mu_:.2f}\nStd Dev: {std_:.2f}", verticalalignment='top', horizontalalignment='right', transform=plt.gca().transAxes, fontsize=15, bbox=dict(facecolor='white', alpha=0.5))

# Add vertical lines for mean and mean +/- std
plt.axvline(mu_, color='blue', linestyle='-', linewidth=1.5)  # Blue solid line for mean
plt.axvline(mu_ + std_, color='purple', linestyle='-.', linewidth=1.5)  # Purple dash-dot line for mean + std
plt.axvline(mu_ - std_, color='purple', linestyle='-.', linewidth=1.5)  # Purple dash-dot line for mean - std

plt.xlabel(r'Difference in $r - H\alpha$ color (S-PLUS - VPHAS+)', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.legend(['Gaussian Fit', 'Mean', 'Mean ± Std Dev'], loc='upper right', fontsize=15)

plt.savefig("Figs/comparison_SPLUS_VPHAS_rj660_with_gaussian_fit.pdf")
