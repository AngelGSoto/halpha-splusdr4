'''
This script makes the Lamost spectra overlapped with the SPLUS photometry
'''
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord 
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
import seaborn as sns
import glob
import argparse
import sys
import os
import pandas as pd
from astropy.visualization import hist
from astroML.datasets import fetch_imaging_sample, fetch_sdss_S82standards
from astroML.crossmatch import crossmatch_angular

# Read the file
parser = argparse.ArgumentParser(
    description="""Make a spectra""")

parser.add_argument("fileLamost", type=str,
                    default="teste-program",
                    help="Name of file, taken the prefix")

parser.add_argument("TableSplus", type=str,
                    default="teste-program",
                    help="Name of table, taken the prefix")

parser.add_argument("--ymin", required=False, type=float, default=None,
                    help="""Value y-axis min""")

parser.add_argument("--ymax", required=False, type=float, default=None,
                    help="""Value y-axis max""")

cmd_args = parser.parse_args()
file_spec = cmd_args.fileLamost + ".fits"
file_table = cmd_args.TableSplus + ".ecsv"

hdu = fits.open(file_spec)

datadir = "../"
try:
    table = Table.read(os.path.join(datadir, file_table), format="ascii.ecsv")
except FileNotFoundError:
    file_table = cmd_args.TableSplus + ".csv"
    df = pd.read_csv(os.path.join(datadir, file_table))
    table = Table.from_pandas(df)

# Coordinates of the Lamost
ra = hdu[0].header["RA"]
dec = hdu[0].header["DEC"]
lmX = np.empty((1, 2), dtype=np.float64)
lmX[:, 0] = ra
lmX[:, 1] = dec

# Put in array type Splus table cor
ra1 = table['RA']
dec1 = table['DEC']
spX = np.array(list(zip(ra1, dec1)))

# Find the Lamost object on the SPLUS list
max_radius = 2. / 3600  # 2 arcsec
dist, ind = crossmatch_angular(lmX, spX, max_radius)
match = ~np.isinf(dist)

dist_match = dist[match]
dist_match *= 3600

print("******************************************************")
print("Coordinate Lamost source:", lmX)
print("Coordinate Splus source:", spX[ind])
print("******************************************************")

# Data from the lamost spectra
hdudata = hdu[0].data
wl = hdudata[2]
Flux = hdudata[0]

# Data of the SPLUs list
mag_br, mag_err_br, mag_nr, mag_err_nr = [], [], [], []

wl_br = [3485, 4803,  6250,  7660,  9110]
wl_nr = [3785, 3950, 4100, 4300, 5150,  6600, 8610]
color_br = ["#CC00FF", "#006600",  "#FF0000", "#990033",  "#330034"]
color_nr = ["#9900FF", "#6600FF", "#0000FF", "#009999",  "#DD8000",  "#CC0066", "#660033"]
marker_br = ["s", "s",  "s", "s", "s"]
marker_nr = ["o", "o", "o", "o", "o",  "o", "o"]

mag_br.append(table["u_PStotal"][ind]) 
mag_nr.append(table["J0378_PStotal"][ind])
mag_nr.append(table["J0395_PStotal"][ind])
mag_nr.append(table["J0410_PStotal"][ind])
mag_nr.append(table["J0430_PStotal"][ind])
mag_br.append(table["g_PStotal"][ind])
mag_nr.append(table["J0515_PStotal"][ind]) 
mag_br.append(table["r_PStotal"][ind]) 
mag_nr.append(table["J0660_PStotal"][ind])
mag_br.append(table["i_PStotal"][ind]) 
mag_nr.append(table["J0861_PStotal"][ind]) 
mag_br.append(table["z_PStotal"][ind])

#ERRO PStotal
mag_err_br.append(float(table["e_u_PStotal"][ind]))
mag_err_nr.append(float(table["e_J0378_PStotal"][ind]))
mag_err_nr.append(float(table["e_J0395_PStotal"][ind]))
mag_err_nr.append(float(table["e_J0410_PStotal"][ind]))
mag_err_nr.append(float(table["e_J0430_PStotal"][ind]))
mag_err_br.append(float(table["e_g_PStotal"][ind]))
mag_err_nr.append(float(table["e_J0515_PStotal"][ind])) 
mag_err_br.append(float(table["e_r_PStotal"][ind])) 
mag_err_nr.append(float(table["e_J0660_PStotal"][ind])) 
mag_err_br.append(float(table["e_i_PStotal"][ind]))
mag_err_nr.append(float(table["e_J0861_PStotal"][ind]))
mag_err_br.append(float(table["e_z_PStotal"][ind]))

# Find scale factor
m = wl == 6250.289
wl_part = wl[m]
flux_part = Flux[m]
Fsp = (10**(-(table["r_PStotal"][ind] + 2.41) / 2.5)) / 6250.0**2
factor = flux_part / Fsp

# Using other wl for find the factor of scale
m1 = wl == 6250.3125
wl_part1 = wl[m1]
flux_part1 = Flux[m1]
Fsp1 = (10**(-(table["r_PStotal"][ind] + 2.41) / 2.5)) / 6250.0**2
factor1 = flux_part1 / Fsp1

# Propagation of error
err_br = []
for wll, magg, magerr in zip(wl_br, mag_br, mag_err_br):
    c = (10**(-2.41/2.5)) / wll**2
    b = -(1 / 2.5)
    err = np.sqrt(((c*10**(b*magg))**2)*(np.log(10)*b*magerr)**2)
    err_br.append(float(err))  # Ensure err is converted to a float

err_nr = []
for wll, magg, magerr in zip(wl_nr, mag_nr, mag_err_nr):
    c = (10**(-2.41/2.5)) / wll**2
    b = -(1 / 2.5)
    err = np.sqrt(((c*10**(b*magg))**2)*(np.log(10)*b*magerr)**2)
    err_nr.append(float(err))  # Ensure err is converted to a float

# PLOTS
fig, ax = plt.subplots(figsize=(12, 5))
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.tick_params(axis='x', labelsize=20, width=2, length=8) 
plt.tick_params(axis='y', labelsize=20, width=2, length=8)
ax.set(xlim=[3000, 9300])

# set Y-axis range (if applicable)
if cmd_args.ymin is not None and cmd_args.ymax is not None:
    plt.ylim(cmd_args.ymin, cmd_args.ymax)
elif cmd_args.ymin is not None:
    plt.ylim(ymin=cmd_args.ymin)
elif cmd_args.ymax is not None:
    plt.ylim(ymax=cmd_args.ymax)

ax.set_xlabel('Wavelength $(\AA)$', fontsize=20)
ax.set_ylabel('Relative flux', fontsize=20)

ax.plot(wl, Flux, c="#808080", linewidth=1.5, alpha=0.6, zorder=2)

for wl1, mag, magErr, colors, marker_ in zip(wl_br, mag_br, err_br, color_br, marker_br):
    F = (10**(-(mag + 2.41) / 2.5)) / wl1**2
    try:
        F *= factor
        magErr *= factor
    except ValueError:
        F *= factor1
        magErr *= factor1
    ax.scatter(wl1, F, color=colors, marker=marker_, edgecolors='k', s=200, zorder=4)
    ax.errorbar(wl1, F, yerr=magErr, fmt='o', linestyle=(0, (5, 5)), ecolor=colors, elinewidth=3.9, markeredgewidth=3.2, capsize=10)

for wl1, mag, magErr, colors, marker_ in zip(wl_nr, mag_nr, err_nr, color_nr, marker_nr):
    F = (10**(-(mag + 2.41) / 2.5)) / wl1**2
    try:
        F *= factor
        magErr *= factor
    except ValueError:
        F *= factor1
        magErr *= factor1
    ax.scatter(wl1, F, color=colors, marker=marker_, edgecolors='k', s=180, zorder=4)
    ax.errorbar(wl1, F, yerr=magErr, fmt='o', linestyle=(0, (5, 5)), ecolor=colors, elinewidth=3.9, markeredgewidth=3.2, capsize=10)

# Add minor tick locators without showing the minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(200))  # Adjust interval as needed
ax.yaxis.set_minor_locator(MultipleLocator(0.1))   # Adjust interval as needed
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

plt.tight_layout()

for i in table[ind]:
    if i["ID"].endswith(" '"):
        asciifile = file_spec.replace(".fits", "-"+i["ID"].split("R3.")[-1].split(" ")[0].replace(".", "-")+".pdf")
    else:
        asciifile = file_spec.replace(".fits", "-"+i["ID"].split("R3.")[-1].split("'")[0].replace(".", "-")+".pdf")

plt.savefig(asciifile)
