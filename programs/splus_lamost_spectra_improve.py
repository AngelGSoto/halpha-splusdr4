import argparse
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astroML.crossmatch import crossmatch_angular
import matplotlib.pyplot as plt

def read_table(file_path):
    """Read the input table file."""
    _, file_extension = os.path.splitext(file_path)
    if file_extension == ".ecsv":
        return Table.read(file_path, format="ascii.ecsv")
    elif file_extension == ".csv":
        return Table.read(file_path, format="ascii.csv")
    else:
        raise ValueError("Unsupported file format. Only ECSV and CSV are supported.")

def match_coordinates(file_spec, file_table):
    """Match coordinates between Lamost spectra and SPLUS photometry."""
    hdu = fits.open(file_spec)
    table = read_table(file_table)

    ra = hdu[0].header.get("RA")
    dec = hdu[0].header.get("DEC")
    lmX = np.array([[ra, dec]])

    ra1 = table["RA"]
    dec1 = table["DEC"]
    spX = np.column_stack((ra1, dec1))

    max_radius = 2. / 3600  # 2 arcsec
    dist, ind = crossmatch_angular(lmX, spX, max_radius)
    match = ~np.isinf(dist)

    dist_match = dist[match]
    dist_match *= 3600

    print("******************************************************")
    print("Coordinate Lamost source:", lmX)
    print("Coordinate Splus source:", spX[ind])
    print("******************************************************")

    return table, ind

def main(fileLamost, TableSplus, ymin=None, ymax=None):
    """Main function to generate plots."""
    file_spec = fileLamost + ".fits"
    file_table = TableSplus + ".csv"

    table, ind = match_coordinates(file_spec, file_table)

    hdu = fits.open(file_spec)
    hdudata = hdu[0].data
    wl = hdudata[2]
    Flux = hdudata[0]

    # Scale factor calculation
    m = np.where(wl == 6250.289)[0]
    if len(m) > 0:
        scale_factor = Flux[m] / ((10**(-(table["r_PStotal"][ind] + 2.41) / 2.5)) / 6250.0**2)
    else:
        m1 = np.where(wl == 6250.3125)[0]
        scale_factor = Flux[m1] / ((10**(-(table["r_PStotal"][ind] + 2.41) / 2.5)) / 6250.0**2)

    # Propagation of error
    err_br = np.array([np.sqrt(((10**(-2.41/2.5) / wl_br**2 * 10**(-0.4 * mag_br))**2 * (np.log(10) / 2.5 * mag_err_br))**2).tolist() for wl_br, mag_br, mag_err_br in zip([3485, 4803, 6250, 7660, 9110], table["u_PStotal", "g_PStotal", "r_PStotal", "i_PStotal", "z_PStotal"][ind], table["e_u_PStotal", "e_g_PStotal", "e_r_PStotal", "e_i_PStotal", "e_z_PStotal"][ind])])
    err_nr = np.array([np.sqrt(((10**(-2.41/2.5) / wl_nr**2 * 10**(-0.4 * mag_nr))**2 * (np.log(10) / 2.5 * mag_err_nr))**2).tolist() for wl_nr, mag_nr, mag_err_nr in zip([3785, 3950, 4100, 4300, 5150, 6600, 8610], table["J0378_PStotal", "J0395_PStotal", "J0410_PStotal", "J0430_PStotal", "J0515_PStotal", "J0660_PStotal", "J0861_PStotal"][ind], table["e_J0378_PStotal", "e_J0395_PStotal", "e_J0410_PStotal", "e_J0430_PStotal", "e_J0515_PStotal", "e_J0660_PStotal", "e_J0861_PStotal"][ind])])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set(xlim=[3000, 9700])

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    ax.set_xlabel('Wavelength $(\AA)$', fontsize=32)
    ax.set_ylabel('Normalized flux', fontsize=32)

    ax.plot(wl, Flux, c="gray", linewidth=1.3, alpha=0.6, zorder=5)

    for wl_br, mag_br, mag_err_br, color_br in zip([3485, 4803, 6250, 7660, 9110], table["u_PStotal", "g_PStotal", "r_PStotal", "i_PStotal", "z_PStotal"][ind], table["e_u_PStotal", "e_g_PStotal", "e_r_PStotal", "e_i_PStotal", "e_z_PStotal"][ind], ["#CC00FF", "#006600", "#FF0000", "#990033", "#330034"]):
        F = (10**(-(mag_br + 2.41) / 2.5)) / wl_br**2 * scale_factor
        ax.scatter(wl_br, F, color=color_br, marker='s', facecolors="none", s=200, zorder=4)
        ax.errorbar(wl_br, F, yerr=mag_err_br * scale_factor, fmt='r', marker=None, linestyle=(0, (5, 5)), color=color_br, ecolor=color_br, elinewidth=3.9, markeredgewidth=3.2, capsize=10)

    for wl_nr, mag_nr, mag_err_nr, color_nr in zip([3785, 3950, 4100, 4300, 5150, 6600, 8610], table["J0378_PStotal", "J0395_PStotal", "J0410_PStotal", "J0430_PStotal", "J0515_PStotal", "J0660_PStotal", "J0861_PStotal"][ind], table["e_J0378_PStotal", "e_J0395_PStotal", "e_J0410_PStotal", "e_J0430_PStotal", "e_J0515_PStotal", "e_J0660_PStotal", "e_J0861_PStotal"][ind], ["#9900FF", "#6600FF", "#0000FF", "#009999", "#DD8000", "#CC0066", "#660033"]):
        F = (10**(-(mag_nr + 2.41) / 2.5)) / wl_nr**2 * scale_factor
        ax.scatter(wl_nr, F, color=color_nr, marker='o', s=180, zorder=4)
        ax.errorbar(wl_nr, F, yerr=mag_err_nr * scale_factor, fmt='r', marker=None, linestyle=(0, (5, 5)), color=color_nr, ecolor=color_nr, elinewidth=3.9, markeredgewidth=3.2, capsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Make a spectras""")
    parser.add_argument("fileLamost", type=str, default="teste-program", help="Name of file, taken the prefix")
    parser.add_argument("TableSplus", type=str, default="teste-program", help="Name of table, taken the prefix")
    parser.add_argument("--ymin", required=False, type=float, default=None, help="""Value y-axis min""")
    parser.add_argument("--ymax", required=False, type=float, default=None, help="""Value y-axis max""")
    cmd_args = parser.parse_args()

    main(cmd_args.fileLamost, cmd_args.TableSplus, ymin=cmd_args.ymin, ymax=cmd_args.ymax)
