from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
import seaborn as sns
sns.set_theme(style="ticks")

# Define colors and symbols
color_rrlyrae = '#377eb8'  # Dark blue for RR Lyrae stars
border_color_rrlyrae = 'black'  # Black border for better contrast

# Load data
df = pd.read_csv("RRlyrae/Tab_2_J_A+A_607_A11_table1-SPLUS_ID_clean-moreParameters.csv")

# Calculate colors
ri = df["r_PStotal"] - df["i_PStotal"]
rj660 = df["r_PStotal"] - df["J0660_PStotal"]

# Plotting Halpha-colorDistribution
fig, ax = plt.subplots(figsize=(14, 11))
plt.xlabel(r"$r - i$", fontsize=28)
plt.ylabel(r"$r - J0660$", fontsize=28)
plt.tick_params(axis='both', labelsize=28, width=3, length=12)

# Scatter plot for RR Lyrae stars
scatter_rrlyrae = ax.scatter(
    ri, rj660,
    color=color_rrlyrae,
    alpha=0.8,
    s=300,  # Size of points
    marker='*',  # Star symbol for RR Lyrae
    edgecolors=border_color_rrlyrae,
    linewidths=1.9,
    zorder=2, label="RR Lyrae"
)

# Add horizontal line at r - J0660 = 0.0
ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=2, label='r - J0660 = 0.0')

# Set limits for axes
ax.set_xlim(-0.5, 0.6)
ax.set_ylim(-0.3, 0.7)

# Add minor tick locators without showing the minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_formatter(NullFormatter())

# Add legend
# ax.legend(loc='upper left', fontsize=35)

# Add text with match information
ax.text(0.5, 0.95, 'RR Lyrae matched with S-PLUS\n(Greer et al. 2017)',
        transform=ax.transAxes, fontsize=25, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

plt.tight_layout()    
plt.savefig("Figs/RRlyrae-colorDistribution.pdf")
