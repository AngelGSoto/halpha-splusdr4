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

# Iterative fitting function
def iterative_fit(x, y, niter=2):
    # Initial fit
    fit = fitting.LinearLSQFitter()
    line_init = models.Linear1D()
    fitted_line = fit(line_init, x, y)
    
    # Iteratively force the fit upwards
    for i in range(niter):
        # Select only the objects above the fitted line
        mask = y >= fitted_line(x)
        x_fit = x[mask]
        y_fit = y[mask]
        
        # Fit the selected points
        fitted_line = fit(line_init, x_fit, y_fit)
    
    return fitted_line


# Read the file
parser = argparse.ArgumentParser(
    description="""This script selects Halpha emitters based on color-color diagrams. 
                   You can specify the filename, and variance estimation method.""")

parser.add_argument("fileName", type=str,
                    default="teste-program",
                    help="Name of table, taken the prefix ")

parser.add_argument("--Ranger", type=str,
                    default="r < 16",
                    help="Base name of FITS image that contains the source")

parser.add_argument("--varianceApproach", type=str, choices=["Maguio", "Mine", "Fratta"], default="Fratta",
                    help="Approach for estimating variance (choose from 'Manguio', 'Mine', 'Fratta').")

parser.add_argument("--niterFile", type=str, default="niter_table.dat",
                    help="File containing the number of iterations needed for each field.")

cmd_args = parser.parse_args()
file_ = cmd_args.fileName + ".csv"
niter_file = cmd_args.niterFile

datadir = "Disk_Lomeli/"
datadir1 = "Disk_fields_method_tofit/"

# Load iteration data
try:
    niter_df = pd.read_csv(os.path.join(datadir1, niter_file), delim_whitespace=True)
except FileNotFoundError:
    niter_df = pd.read_csv(niter_file, delim_whitespace=True)

# Load main data
try:
    df = pd.read_csv(os.path.join(datadir, file_))
except FileNotFoundError:
    df = pd.read_csv(file_)

print("Number of stars=", len(df))
# Creating the color to creating the diagram from IPHAS
# Grouping the data by field
grouped_df = df.groupby("field")
ha_emitter_data = []  # To store the Halpha emitter data for each field

# Loop through each field
for field, data_ in grouped_df:
    cx, cy = colour(data_, "r", "i", "r", "F660")

    # Determine the number of iterations for this field
    niter_row = niter_df[niter_df['Field'] == field]['niter'].values
    niter = int(niter_row[0]) if len(niter_row) > 0 and niter_row[0] != 'n' else 0  # 0 means no iterative fitting

    if niter > 0:
        # Perform iterative fitting
        fitted_line = iterative_fit(cx, cy, niter=niter)
    else:
        # Perform basic linear fitting if no iteration is needed
        fit = fitting.LinearLSQFitter()
        line_init = models.Linear1D()
        # Initialize the outlier removal fitter
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=4, sigma=4.0)
        # Fit the data with the fitter and sigma clipping
        fitted_line, mask = or_fit(line_init, cx, cy)

    cy_predic = fitted_line(cx)

    # Get the coefficients of the fitted line after iterative fitting
    a_iter = fitted_line.slope.value
    b_iter = fitted_line.intercept.value
    intercept_str_iter = f"{b_iter:.2f}" if b_iter >= 0 else f"- {-b_iter:.2f}"
    print(f"Field: {field},", f'$y = {a_iter:.2f}x {intercept_str_iter}$')
    

    # Create DataFrame with the new columns
    colum1 = pd.DataFrame(cx, columns=['r - i'])
    colum2 = pd.DataFrame(cy, columns=['r - J0660'])
    data = pd.concat([data_["ALPHA"], data_["DELTA"], df["fwhm_r"], data_["r"], colum1, colum2], axis=1)

    # Estimating parameter for statistical
    residuals = cy - cy_predic
    sigma_fit = np.std(residuals)

    # Errors on the colors
    ecx, ecy = errormag(data_, "rerr", "ierr", "rerr", "F660err")

    # Create DataFrame with errors
    colum_ri = pd.DataFrame(ecx, columns=['e(r - i)'])
    colum_rh = pd.DataFrame(ecy, columns=['e(r - J0660)'])
    data_final = pd.concat([data, colum_ri, colum_rh], axis=1)

    # Selecting the H𝛼 emitters
    m = fitted_line.slope  # Slope of the fit line
    C = 5.0  # Constant

    if cmd_args.varianceApproach == "Maguio":
        variance_est = sigma_fit**2 + m**2 * data_["ierr"]**2 + (1 - m)**2 * data_["rerr"]**2 + data_["F660err"]**2
    elif cmd_args.varianceApproach == "Mine":
        variance_est = sigma_fit**2 + m**2 * data_final["e(r - i)"]**2 + (1 - m)**2 * data_final["e(r - J0660)"]**2
    else:  # Default to Fratta
        variance_est = sigma_fit**2 + m**2 * data_final["e(r - i)"]**2 + data_final["e(r - J0660)"]**2

    criterion = C * np.sqrt(variance_est)
    
    cy = pd.Series(cy, index=data_["ALPHA"].index)
    cy_predic_sigma_clip = pd.Series(cy_predic, index=data_["ALPHA"].index)
    criterion = pd.Series(criterion, index=data_["DELTA"].index)

    mask_ha_emitter = (cy - cy_predic) >= criterion

    # Filter original data based on the mask
    data_ha_emitter_orig = data_[mask_ha_emitter]

    # Append filtered original data to the list
    ha_emitter_data.append(data_ha_emitter_orig)

# Save the resulting table
ha_emitter_combined = pd.concat(ha_emitter_data, ignore_index=True)

df_file = "Ha-emitters-disk-iteractive/Halpha-{}-{}_PerField.csv".format(file_.split('.cs')[0], cmd_args.varianceApproach) 
ha_emitter_combined.to_csv(df_file, index=False)

asciifile = "Ha-emitters-disk-iteractive/Halpha-{}-{}_PerField.ecsv".format(file_.split('.cs')[0], cmd_args.varianceApproach) 
Table.from_pandas(ha_emitter_combined).write(asciifile, format="ascii.ecsv", overwrite=True)
