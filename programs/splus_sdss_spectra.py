'''
This script makes the SDSS spectra overlapped on SPLUS photometry
'''
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord 
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
import seaborn as sn
import glob
import argparse
import sys
import os
from astropy.visualization import hist
from astroML.datasets import fetch_imaging_sample, fetch_sdss_S82standards
from astroML.crossmatch import crossmatch_angular

# Read the files
parser = argparse.ArgumentParser(
    description="""Make spectra""")

parser.add_argument("fileSdss", type=str,
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
file_spec = cmd_args.fileSdss + ".fits"
file_table = cmd_args.TableSplus + ".ecsv"

# SDSS spectra
datadir_sdss = "SDSS-spectra/"
try:
    hdu = fits.open(file_spec)
except FileNotFoundError:
    hdu = fits.open(os.path.join(datadir_sdss, file_spec))

# Table
datadir = "../"

try:
    table = Table.read(os.path.join(datadir, file_table), format="ascii.ecsv")
except FileNotFoundError:
    file_table = cmd_args.TableSplus + ".csv"
    df = pd.read_csv(os.path.join(datadir, file_table))
    table = Table.from_pandas(df)
    
#table = Table.from_pandas(df)
# Coordinates of the SDSS
ra = hdu[0].header["PLUG_RA"]
dec = hdu[0].header["PLUG_DEC"]
sdX = np.empty((1, 2), dtype=np.float64)
sdX[:, 0] = ra
sdX[:, 1] = dec

# Put in array type Splus table coordinate
ra1 = table['RA']
dec1 = table['DEC']
spX = np.array(list(zip(ra1, dec1)))

# Find the SDSS object on the SPLUS list, makes crossmacth using 2 arcsec 
max_radius = 2. / 3600  # 2 arcsec
dist, ind = crossmatch_angular(sdX, spX, max_radius)
match = ~np.isinf(dist)

dist_match = dist[match]
dist_match *= 3600

print("******************************************************")
print("Coordinate SDSS source:", sdX)
print("Coordinate SPLUS source:", spX[ind])
print("******************************************************")

# Data from the SDSS spectra
hdudata = hdu[1].data
wl = 10**hdudata.field("loglam")
Flux = 1E-17*hdudata.field("flux")

# Data of the SPLUs list
mag_br, mag_err_br, mag_nr, mag_err_nr = [], [], [], []
#wl_sp = [3485, 3785, 3950, 4100, 4300, 4803, 5150, 6250, 6600, 7660, 8610, 9110]
#color = ["#CC00FF", "#9900FF", "#6600FF", "#0000FF", "#009999", "#006600", "#DD8000", "#FF0000", "#CC0066", "#990033", "#660033", "#330034"]
#marker = ["s", "o", "o", "o", "o", "s", "o", "s", "o", "s", "o", "s"] ### tienen todos los filtros

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

# Propagation of error
err_br = []
for wll, magg, magerr in zip(wl_br, mag_br, mag_err_br):
    c = (10**(-2.41/2.5)) / wll**2
    c /= 1e-15
    b = -(1. / 2.5)
    err = np.sqrt(((c*10**(b*magg))**2)*(np.log(10)*b*magerr)**2)
    err_br.append(err)

err_nr = []
for wll_nr, magg_nr, magerr_nr in zip(wl_nr, mag_nr, mag_err_nr):
    c_nr = (10**(-2.41/2.5)) / wll_nr**2
    c_nr /= 1e-15
    b_nr = -(1. / 2.5)
    err_nr_ = np.sqrt(((c_nr*10**(b_nr*magg_nr))**2)*(np.log(10)*b_nr*magerr_nr)**2)
    err_nr.append(err_nr_)


# Selecting one emission line from the dictionary
# CV
# selected_emission_line = "Hα"  # Change this line to select a different emission line
# z = 0
Type = table["main_type"][ind][0]
z = table["redshift"][ind]

if 3.2 <= z <= 3.4:
    selected_emission_line = "C IV 1551"
    offset_x = 200
    offset_yy = 0.1
    offset_y = 0.035
elif 2.4 <= z <= 2.55:
    selected_emission_line = "C III] 1909"
    offset_x = 200
    offset_yy = 0.3
    offset_y = 0.055
elif 1.3 <= z <= 1.4:
    selected_emission_line = "Mg II 2799"
    offset_x = 200
    offset_yy = 0.3
    offset_y = 0.055
elif  0.33 <= z <= 0.4: 
    selected_emission_line = "[O III] 5007"
    offset_x = 200
    offset_yy = 0.1
    offset_y = 0.035
else:
    selected_emission_line = "Hα"
    z=0
    offset_x = 0
    offset_yy = 0
    offset_y = 0.05

   
# Calculate max flux around the selected emission line for label positioning
emission_lines = {
    "Lyα": 1215.670,
    "C IV 1551": 1550.772,
    "C III] 1909": 1908.734,
    "Mg II 2799": 2799,
    "Hγ": 4340.471, 
    "[O III] 4363": 4363.21,
    "He I 4472": 4471.5,
    "Hβ": 4861.33,
    "[O III] 4959": 4958.911,
    "[O III] 5007": 5006.843,
    "He I 5876": 5875.66,    
    "[O I] 6300": 6300.3,
    "[S III] 6312": 6312.1,
    "[O I] 6364": 6363.77,
    "Hα": 6562.82,
    "[N II] 6584": 6583.50,
    "He I 6678": 6678.16,
    "[S II] 6716": 6716.44,
    "[S II] 6731": 6730.82,
    "He I 7065": 7065.25,
    "[Ar III] 7136": 7135.80,
    "[O II] 7319": 7319.45,
    "[O II] 7330": 7330.20,
}

selected_wavelength = emission_lines[selected_emission_line]
lambda_ob = selected_wavelength * (z + 1)
j = lambda_ob - 10
k = lambda_ob + 10
mask = (j < wl) & (wl < k)
flux_values = Flux[mask]
flux_values /=1e-15
flux_values +=0.05
max_flux = np.max(flux_values) if len(flux_values) > 0 else 10

# PLOTS
fig, ax = plt.subplots(figsize=(12, 5))
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
plt.tick_params(axis='x', labelsize=20, width=2, length=8) 
plt.tick_params(axis='y', labelsize=20, width=2, length=8)
ax.set(xlim=[3000, 9300])

#axis limit
mask_lim = (wl > 6100.) & (wl < 6900.)
Flux_lim = Flux[mask_lim]
if max(Flux_lim) > 5 * np.mean(Flux_lim):
    max_y_lim = max(Flux_lim) * .9

# set Y-axis range (if applicable)
if cmd_args.ymin is not None and cmd_args.ymax is not None:
    plt.ylim(cmd_args.ymin,cmd_args.ymax)
elif cmd_args.ymin is not None:
    plt.ylim(ymin=cmd_args.ymin)
elif cmd_args.ymax is not None:
    plt.ylim(ymax=cmd_args.ymax)

# set labels and font size
ax.set_xlabel('Wavelength $(\AA)$', fontsize = 20)
ax.set_ylabel(r'F$(\mathrm{10^{-15} erg\ s^{-1} cm^{-2} \AA^{-1}})$', fontsize = 20)
Flux /=1e-15

ax.plot(wl, Flux, c="#808080", linewidth=1.5, alpha=0.6, zorder=2)
for wl1, mag, magErr, colors, marker_ in zip(wl_br, mag_br, err_br, color_br, marker_br): #
    F = (10**(-(mag + 2.41) / 2.5)) / wl1**2
    F /= 1e-15
    ax.scatter(wl1, F, color=colors, marker=marker_, edgecolors='k', s=200, zorder=4)
    ax.errorbar(wl1, F, yerr=magErr, fmt='r', marker=None, linestyle=(0, (5, 5)), color=colors, ecolor=colors, elinewidth=3.9, markeredgewidth=3.2, capsize=10)

for wl1, mag, magErr, colors, marker_ in zip(wl_nr, mag_nr, err_nr, color_nr, marker_nr):
    F = (10**(-(mag + 2.41) / 2.5)) / wl1**2
    F /= 1e-15
    ax.scatter(wl1, F, color=colors, marker=marker_, edgecolors='k', s=180, zorder=4)
    ax.errorbar(wl1, F, yerr=magErr, fmt='r', marker=None, linestyle=(0, (5, 5)), color=colors, ecolor=colors, elinewidth=3.9, markeredgewidth=3.2, capsize=10)

# Putting the labels to the lines
# Putting the label for the selected emission line
ax.axvline(lambda_ob, color='k', linewidth=0.9, alpha=0.9, linestyle='--')
bbox_props = dict(boxstyle="round", fc="w", ec="0.88", alpha=0.6, pad=0.1)
ax.annotate(selected_emission_line, (lambda_ob, max_flux), alpha=1, size=18,
            xytext=(10.5, 5.6), textcoords='offset points', ha='right', va='bottom',
            rotation=90, bbox=bbox_props, zorder=200)

ax.annotate(Type, (8500, max_flux+offset_yy), alpha=1, size=18,
            xytext=(7.5, 5.6), textcoords='offset points', ha='right', va='bottom',
            rotation=0, bbox=bbox_props, zorder=200)

ax.annotate("z = {:.3f}".format(float(z)), (8500+offset_x, max_flux+offset_yy-offset_y), alpha=1, size=18,
            xytext=(7.5, 5.6), textcoords='offset points', ha='right', va='bottom',
            rotation=0, bbox=bbox_props, zorder=200)

# Add minor tick locators without showing the minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(200))  # Adjust interval as needed
ax.yaxis.set_minor_locator(MultipleLocator(0.1))   # Adjust interval as needed
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

plt.tight_layout()
for i in table[ind]:
    if i["ID"].endswith(" '"):
        asciifile = file_spec.replace(".fits", 
                  "-"+i["ID"].split("R3.")[-1].split(" ")[0].replace(".", "-")+".pdf")
    else:
        asciifile = file_spec.replace(".fits", 
                  "-"+i["ID"].split("R3.")[-1].split("'")[0].replace(".", "-")+".pdf")
#path_save = "../SDSS-spectra-paper/"
#path_save = "../../paper/Figs2/"
#plt.savefig(os.path.join(path_save, asciifile))
plt.savefig(asciifile)
