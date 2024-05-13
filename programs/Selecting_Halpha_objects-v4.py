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


# Creating the color to creating the diagram from IPHAS
cx, cy = colour(df, "r_PStotal", "i_PStotal", "r_PStotal", "J0660_PStotal")

# 𝐔𝐬𝐢𝐧𝐠 𝐚𝐬𝐭𝐫𝐨𝐩𝐲 𝐭𝐨 𝐝𝐨 𝐭𝐡𝐞 𝐟𝐢𝐭 𝐥𝐢𝐧𝐞

# Initialize a linear fitter
fit = fitting.LinearLSQFitter()

# Initialize a linear model
line_init = models.Linear1D()

# Fit the data with the fitter
fitted_line = fit(line_init, cx, cy)

print("Original fitted model", fitted_line)

# Initialize the outlier removal fitter
or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=4, sigma=4.0)

# Fit the data with the fitter
fitted_line_, mask = or_fit(line_init, cx, cy)
filtered_data = np.ma.masked_array(cy, mask=mask)

print("Iterative fitted model with iterative sigma clipped ", fitted_line_)

# Create DataFrame with the new colums
colum1 = pd.DataFrame(cx, columns=['r - i'])
colum2 = pd.DataFrame(cy, columns=['r - J0660'])
data = pd.concat([df["RA"], df["DEC"], df["FWHM"], df["r_PStotal"], colum1, colum2], axis=1)

# Estimating parameter for statistical
cy_predic = fitted_line_(data['r - i'])
sigma_fit = mean_squared_error(cy, cy_predic, squared=False)
print("The root mean squared value of the residuals around the fit:", sigma_fit)

################################################################################################################
# Selecting the alpha emitters
# Applying the selection criteria to selecting H𝛼 emitters. We used the same procedure in Wevers et al. (2017).
# The objects with H𝛼 excess meet the condition:

# (𝑟−𝐽0660)𝑜𝑏𝑠−(𝑟−𝐽0660)𝑓𝑖𝑡≥𝐶×𝜎2𝑠−𝜎2𝑝ℎ𝑜𝑡⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯√

# where 𝜎𝑠 is the root mean squared value of the residuals around the fit and 𝜎𝑝ℎ𝑜𝑡 is the error on the observed (𝑟−𝐽0660) colour

# First see an aproximation of the 4𝜎 cut away from the ariginal fit.
ecx, ecy = errormag(df, "e_r_PStotal", "e_i_PStotal", "e_r_PStotal", "e_J0660_PStotal")

# Create DataFrame with the new colums with the errors on the colours
colum_ri = pd.DataFrame(ecx, columns=['e(r - i)'])
colum_rh = pd.DataFrame(ecy, columns=['e(r - J0660)'])
data_final = pd.concat([data, colum_ri, colum_rh], axis=1)

# Applying the criterion 
C = 5.0 # Is the constant
crite = C * np.sqrt(sigma_fit**2 + data_final["e(r - J0660)"]**2) + cy_predic
mask_ha_emitter = data_final["r - J0660"] >= crite
# Applying mask to the data
data_ha_emitter  = data_final[mask_ha_emitter]


x_values = np.linspace(-5.0, 5.0)
#################################################################################################################
# Plots
#################################################################################################################
fontP = FontProperties()
fontP.set_size('xx-large')
color_palette = sns.color_palette('Paired', 12)
with sns.axes_style('white'):
    fig, ax1 = plt.subplots(figsize=(16, 13))
    ax1.spines["top"].set_visible(False)  
    ax1.spines["right"].set_visible(False)
    plt.xlabel(r"$r - i$", fontsize=35)
    plt.ylabel(r"$r - J0660$", fontsize=35)
    plt.tick_params(axis='x', labelsize=35) 
    plt.tick_params(axis='y', labelsize=35)

    scatter = ax1.scatter(
       data['r - i'], data['r - J0660'],
       s=5, 
        color="gray",
       cmap="seismic", alpha = 0.5, zorder=4.0)
    ax1.set(
    xlim=[-2., 3.5],
    ylim=[-1.5, 4.2])
    
    #ax1.plot(fit_line, 0.42744 * fit_line - 0.04264, ls=':', color="k", zorder = 9, label='HDBSCAN Curve_fit')
    #ax1.plot(fit_line, 0.42917 * fit_line - 0.04333, color="k", zorder = 8, label='Curve_fit')
    ax1.plot(x_values, fitted_line(x_values), 'k-', zorder = 6, label='Initial fitted')
    ax1.plot(x_values, fitted_line_(x_values), ls='--', color="k", zorder = 8, label='Iter. fitted $\sigma$ clipped')
    ax1.plot(x_values, fitted_line_(x_values) + 5*sigma_fit, ls=':', color="k", zorder = 8, label='Representation 4$\sigma$')
    plt.fill_between(x_values, fitted_line_(x_values), fitted_line_(x_values) + 4*sigma_fit, color='k', alpha = 0.1)
    ax1.annotate(cmd_args.Ranger, xy=(3.0, 4.2),  xycoords='data', size=25,
            xytext=(-120, -60), textcoords='offset points', 
            bbox=dict(boxstyle="round4,pad=.5", fc="0.9"),)
            #arrowprops=dict(arrowstyle="->",
                            #connectionstyle="angle,angleA=0,angleB=80,rad=20"))

    redde_vector(-1.2314754077697903, 2.147731023789999, -0.8273818571912539, 2.1826566358487645, 4., -2.7, -0.3, -0.15) #E=0.7, this was estimate by comparing the desrending with the redenning model PNe, see Gutiérrez-Soto et al. (2020)
    #representation of the errors
    pro_ri = median(data_final["e(r - i)"])
    pro_rj660 = median(data_final["e(r - J0660)"])
    print("Median", pro_ri)
    axis_coordinates_of_representative_error_bar = (0.78, 0.85)
    screen_coordinates_of_representative_error_bar = ax1.transAxes.transform(axis_coordinates_of_representative_error_bar)
    screen_to_data_transform = ax1.transData.inverted().transform
    data_coordinates_of_representative_error_bar = screen_to_data_transform(screen_coordinates_of_representative_error_bar)
    foo = data_coordinates_of_representative_error_bar

    ax1.errorbar(foo[0], foo[1], xerr=pro_ri, yerr=pro_rj660, c = "k", capsize=3)
    ax1.annotate("Median Errors", xy=(3.15, 3.75),  xycoords='data', size=25,
            xytext=(-120, -60), textcoords='offset points', )
    
    ax1.legend(loc = 'upper left', ncol=1, fontsize=25, title='Fitted models', title_fontsize=30)
    #ax1.set_aspect("equal")
    file_save = "diagram-{}.jpg".format(file_.split('.cs')[0])
    plt.savefig(file_save)

##################################################################################################################
# Save the resultanting table
# Firts merge the orignal table with resulting ones
##################################################################################################################
data_orig_fin = df[mask_ha_emitter]
data_merge = pd.merge(data_orig_fin, data_ha_emitter)

df_file = "Halpha-{}.csv".format(file_.split('.cs')[0]) 
data_merge.to_csv(df_file, index=False)

asciifile = "Halpha-{}.ecsv".format(file_.split('.cs')[0]) 
Table.from_pandas(data_merge).write(asciifile, format="ascii.ecsv", overwrite=True)

