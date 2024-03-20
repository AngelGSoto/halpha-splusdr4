from __future__ import print_function
import numpy as np
from sklearn import metrics
from scipy.optimize import curve_fit
import pandas as pd
from astropy.table import Table
import seaborn as sns
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error
from astropy.modeling import models, fitting
import argparse
import sys
import os
from statistics import median
from pathlib import Path

ROOT_PATH = Path("..")

# Reddening vector
def redde_vector(x0, y0, x1, y1, a, b, c, d):
    plt.arrow(x0+a, y0+b, (x1+a)-(x0+a), (y1+b)-(y0+b),  fc="k", ec="k", width=0.01,
              head_width=0.07, head_length=0.15) #head_width=0.05, head_length=0.1)
    plt.text(x0+a+c, y0+b+d, 'A$_\mathrm{V}=2$', va='center', fontsize=23)

# Define the true objective function
def objective(x, a, b):
    return a * x + b

# Definition for the colors
def colour(table, f1, f2, f3, f4):
    xcolour = table[f1] - table[f2]
    ycolour = table[f3] - table[f4]
    return xcolour, ycolour

# Errors of the colors
def errormag(table, ef1, ef2, ef3, ef4):
    excolour = np.sqrt(table[ef1]**2 + table[ef2]**2)
    eycolour = np.sqrt(table[ef3]**2 + table[ef4]**2)
    return excolour, eycolour

# Read the file
parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("fileName", type=str,
                    default="teste-program",
                    help="Name of table, taken the prefix ")

parser.add_argument("--Ranger", type=str,
                    default="r < 16",
                    help="Base name of FITS image that contains the source")

parser.add_argument("--varianceApproach", type=str, choices=["Maguio", "Mine", "Fratta"], default="Fratta",
                    help="Approach for estimating variance")

cmd_args = parser.parse_args()
file_ = cmd_args.fileName + ".csv"

datadir = "catalogs_bins/"

try:
    df = pd.read_csv(os.path.join(datadir, file_))
except FileNotFoundError:
    df = pd.read_csv(file_)


# Creating the color to creating the diagram from IPHAS
# Grouping the data by field
grouped_df = df.groupby("Field")
ha_emitter_data = []  # To store the Halpha emitter data for each field

# Loop through each field
for field, data_ in grouped_df:
    cx, cy = colour(data_, "r_PStotal", "i_PStotal", "r_PStotal", "J0660_PStotal")

    # 𝐔𝐬𝐢𝐧𝐠 𝐚𝐬𝐭𝐫𝐨𝐩𝐲 𝐭𝐨 𝐝𝐨 𝐭𝐡𝐞 𝐟𝐢𝐭 𝐥𝐢𝐧𝐞

    # Initialize a linear fitter
    fit = fitting.LinearLSQFitter()

    # Initialize a linear model
    line_init = models.Linear1D()

    # Fit the data with the fitter
    fitted_line_normal = fit(line_init, cx, cy)
    cy_predic_normal = fitted_line_normal(cx)

    # Initialize the outlier removal fitter
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=4, sigma=4.0)

    # Fit the data with the fitter and sigma clipping
    fitted_line_sigma_clip, mask = or_fit(line_init, cx, cy)
    cy_predic_sigma_clip = fitted_line_sigma_clip(cx)

    # Create DataFrame with the new columns
    colum1 = pd.DataFrame(cx, columns=['r - i'])
    colum2 = pd.DataFrame(cy, columns=['r - J0660'])
    data = pd.concat([data_["RA"], data_["DEC"], df["FWHM"], data_["r_PStotal"], colum1, colum2], axis=1)

    # Estimating parameter for statistical
    residuals_normal = cy - cy_predic_normal
    sigma_fit_normal = np.std(residuals_normal)

    residuals_sigma_clip = cy - cy_predic_sigma_clip
    sigma_fit_sigma_clip = np.std(residuals_sigma_clip)

    ################################################################################################################
    # Selecting the alpha emitters
    # Applying the selection criteria to selecting H𝛼 emitters. We used the same procedure in Wevers et al. (2017).
    # The objects with H𝛼 excess meet the condition:

    # where 𝜎𝑠 is the root mean squared value of the residuals around the fit and 𝜎𝑝ℎ𝑜𝑡 is the error on the observed (𝑟−𝐽0660) colour

    # First see an approximation of the 4𝜎 cut away from the original fit.
    ecx, ecy = errormag(data_, "e_r_PStotal", "e_i_PStotal", "e_r_PStotal", "e_J0660_PStotal")

    # Create DataFrame with the new columns with the errors on the colours
    colum_ri = pd.DataFrame(ecx, columns=['e(r - i)'])
    colum_rh = pd.DataFrame(ecy, columns=['e(r - J0660)'])
    data_final = pd.concat([data, colum_ri, colum_rh], axis=1)

    # Applying the criterion 
    m = fitted_line_sigma_clip.slope  # Slope of the fit line
    C = 5.0  # Is the constant

    if cmd_args.varianceApproach == "Maguio":
        variance_est = sigma_fit_sigma_clip**2 + m**2 * data_["e_i_PStotal"]**2 + (1 - m)**2 * data_["e_r_PStotal"]**2 + data_["e_J0660_PStotal"]**2
    elif cmd_args.varianceApproach == "Mine":
        variance_est = sigma_fit_sigma_clip**2 + m**2 * data_final["e(r - i)"]**2 + (1 - m)**2 * data_final["e(r - J0660)"]**2
    else:  # Default to Fratta
        variance_est = sigma_fit_sigma_clip**2 + m**2 * data_final["e(r - i)"]**2 +  data_final["e(r - J0660)"]**2

    criterion = C * np.sqrt(variance_est)
    # Ensure Series have the same index before comparison
    cy = pd.Series(cy, index=data_["RA"].index)
    cy_predic_sigma_clip = pd.Series(cy_predic_sigma_clip, index=data_["RA"].index)
    criterion = pd.Series(criterion, index=data_["RA"].index)

    mask_ha_emitter = (cy - cy_predic_sigma_clip) >= criterion
    
    # Filter the original data based on the mask indices
    data_ha_emitter_orig = data_[mask_ha_emitter]

    # Append filtered original data to the list
    ha_emitter_data.append(data_ha_emitter_orig)

##################################################################################################################
# Save the resultanting table
# Firts merge the orignal table with resulting ones
##################################################################################################################
# Concatenate all Halpha emitter data
ha_emitter_combined = pd.concat(ha_emitter_data, ignore_index=True)

df_file = "Halpha-{}-{}_PerField.csv".format(file_.split('.cs')[0], cmd_args.varianceApproach) 
ha_emitter_combined.to_csv(df_file, index=False)

asciifile = "Halpha-{}-{}_PerField.ecsv".format(file_.split('.cs')[0], cmd_args.varianceApproach) 
Table.from_pandas(ha_emitter_combined).write(asciifile, format="ascii.ecsv", overwrite=True)

