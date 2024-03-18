'''
With this script I select Halpha emitters from Splus data.
This algorithm is based on Witham et al. 2006
'''
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

# Reddenign vector
def redde_vector(x0, y0, x1, y1, a, b, c, d):
    plt.arrow(x0+a, y0+b, (x1+a)-(x0+a), (y1+b)-(y0+b),  fc="k", ec="k", width=0.01,
              head_width=0.07, head_length=0.15) #head_width=0.05, head_length=0.1)
    plt.text(x0+a+c, y0+b+d, 'A$_\mathrm{V}=2$', va='center', fontsize=23)

# Read the file
parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("fileName", type=str,
                    default="teste-program",
                    help="Name of table, taken the prefix ")

parser.add_argument("--Ranger", type=str,
                    default="r < 16",
                    help="Base name of FITS image that contains the source")

cmd_args = parser.parse_args()
file_ = cmd_args.fileName + ".csv"

datadir = "catalogs_bins/"

try:
    df = pd.read_csv(os.path.join(datadir, file_))
except FileNotFoundError:
    df = pd.read_csv(file_)

# Function to process each field separately
def process_field(df, field_name):
    # Filter data by field
    field_df = df[df['Field'] == field_name]

    # Creating the color to creating the diagram from IPHAS
    cx, cy = colour(field_df, "r_PStotal", "i_PStotal", "r_PStotal", "J0660_PStotal")

    # Initialize a linear fitter
    fit = fitting.LinearLSQFitter()

    # Initialize a linear model
    line_init = models.Linear1D()

    # Fit the data with the fitter
    fitted_line = fit(line_init, cx, cy)

    # Initialize the outlier removal fitter
    or_fit = fitting.FittingWithOutlierRemoval(model=line_init, fitter=fit, outlier_func=sigma_clip, niter=4, sigma=4.0)

    # Fit the data with the fitter
    fitted_line_, mask = or_fit(cx, cy)
    filtered_data = np.ma.masked_array(cy, mask=mask)

    # Estimating parameter for statistical
    cy_predic = fitted_line_(cx)
    sigma_fit = mean_squared_error(cy, cy_predic, squared=False)

    ################################################################################################################
    # Selecting the alpha emitters
    # Applying the selection criteria to selecting H𝛼 emitters. We used the same procedure in Wevers et al. (2017).
    # The objects with H𝛼 excess meet the condition:

    # (𝑟−𝐽0660)𝑜𝑏𝑠−(𝑟−𝐽0660)𝑓𝑖𝑡≥𝐶×𝜎2𝑠−𝜎2𝑝ℎ𝑜𝑡⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯√

    # where 𝜎𝑠 is the root mean squared value of the residuals around the fit and 𝜎𝑝ℎ𝑜𝑡 is the error on the observed (𝑟−𝐽0660) colour

    # First see an aproximation of the 4𝜎 cut away from the ariginal fit.
    ecx, ecy = errormag(field_df, "e_r_PStotal", "e_i_PStotal", "e_r_PStotal", "e_J0660_PStotal")

    # Applying the criterion 
    C = 4.0 # Is the constant
    crite = C * np.sqrt(sigma_fit**2 + ecy**2) + cy_predic
    mask_ = field_df["r - J0660"] >= crite
    # Applying mask to the data
    data_ha = field_df[mask_]

    return data_ha

# Get unique field names
unique_fields = df['Field'].unique()

# Process each field separately
for field_name in unique_fields:
    data_ha = process_field(df, field_name)
    print("Processed field:", field_name)
    print(data_ha)
