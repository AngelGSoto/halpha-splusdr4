"""
This script just made plot color-color diagram for individual fields
"""
from __future__ import print_function
import pandas as pd
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os
from statistics import median
from matplotlib.colors import PowerNorm
from sklearn.metrics import mean_squared_error
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from pathlib import Path

# Define spectral type ranges for MS stars and giants
MS_star_types = ['o5v', 'o9v', 'b0v', 'b1v', 'b3v', 'b57v', 'b8v', 'b9v',
                 'a0v', 'a2v', 'a3v', 'a5v', 'a7v', 'f0v', 'f2v', 'f5v',
                 'f6v', 'f8v', 'g0v', 'g2v', 'g5v', 'g8v', 'k0v', 'k2v',
                 'k3v', 'k4v', 'k5v', 'k7v', 'm0v', 'm1v', 'm2v', 'm2p5v',
                 'm3v', 'm4v', 'm5v', 'm6v']

giant_types = ['b2ii', 'b5ii', 'a0iii', 'a3iii', 'a5iii', 'a7iii', 'f0iii',
               'f2iii', 'f5iii', 'g0iii', 'g5iii', 'g8iii', 'k0iii', 'k1iii',
               'k2iii', 'k3iii', 'k4iii', 'k5iii', 'm0iii', 'm1iii', 'm2iii',
               'm3iii', 'm4iii', 'm5iii', 'm6iii', 'm7iii', 'm8iii', 'm9iii',
               'm10iii']

def filter_mag(e, s, f1, f2, f3, data):
    '''
    Calculate the colors using any set of filters
    '''
    col, col0 = [], []
    if data['id'].endswith(e):
        prefix = data['id'].split("-")[0]
        if prefix.startswith(str(s)):
            filter1 = data[f1]
            filter2 = data[f2]
            filter3 = data[f3]
            diff = filter1 - filter2
            diff0 = filter1 - filter3
            col.append(diff)
            col0.append(diff0)
    
    return col, col0

def classify_star(prefix):
    """
    Classify the star as MS or Giant based on the spectral type prefix
    """
    if prefix in MS_star_types:
        return 'MS'
    elif prefix in giant_types:
        return 'Giant'
    else:
        return None  # Return None if classification fails

def plot_mag(f1, f2, f3):
    MS_A1, MS_B1, giant_A1, giant_B1 = [], [], [], []
    for file_name in file_list:
        with open(file_name) as f:
            data = json.load(f)
            prefix = data['id'].split("-")[0]
            classification = classify_star(prefix)
            if classification == 'MS':
                x, y = filter_mag("Star", "", f1, f2, f3, data)
                MS_A1.extend(x)
                MS_B1.extend(y)
            elif classification == 'Giant':
                x, y = filter_mag("Star", "", f1, f2, f3, data)
                giant_A1.extend(x)
                giant_B1.extend(y)
    
    return MS_A1, MS_B1, giant_A1, giant_B1

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

parser.add_argument("--Field", type=str,
                    default="b'STRIPE82-0142'",
                    help="Choosing a field")

parser.add_argument("--Object", type=str,
                    default="b'iDR4_3_STRIPE82-0142_0021237'",
                    help="Choosing a individual object in the Field")

parser.add_argument("--Ranger", type=str,
                    default="r < 16",
                    help="Range on r-magnitude")


cmd_args = parser.parse_args()
file_ = cmd_args.fileName + ".csv"

datadir = "catalogs_bins/"

try:
    df = pd.read_csv(os.path.join(datadir, file_))
except FileNotFoundError:
    df = pd.read_csv(file_)

#Choose the field
mask_field = df["Field"] == cmd_args.Field
df_field = df[mask_field]

# Choose the field
mask_obj = df_field["ID"] == cmd_args.Object
df_obj = df_field[mask_obj]

# Remove rows corresponding to the individual object from df_field
df_field = df_field[~mask_obj]


# Creating the color to creating the diagram from IPHAS
cx, cy = colour(df_field, "r_PStotal", "i_PStotal", "r_PStotal", "J0660_PStotal")
# error
ecx, ecy = errormag(df_field, "e_r_PStotal", "e_i_PStotal", "e_r_PStotal", "e_J0660_PStotal")

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

# Get the coefficients of the fitted line after sigma clipping
a = fitted_line_normal.slope.value
b = fitted_line_normal.intercept.value
print("Pendiente:",a)
print("Intercept:",b)

# Check if any rows are selected
if df_obj.empty:
    print(f"No data found for object {cmd_args.Object}. Skipping individual object plot.")
else:
    # Creating the color to creating the diagram from IPHAS
    cx_obj, cy_obj = colour(df_obj, "r_PStotal", "i_PStotal", "r_PStotal", "J0660_PStotal")

# Modify the file pattern to match the JSON files
pattern = "../MS_stars/*.json"
file_list = glob.glob(pattern)


# Plotting

fig, ax = plt.subplots(figsize=(15, 11))
plt.xlabel(r"$r - i$", fontsize=35)
plt.ylabel(r"$r - J0660$", fontsize=35)
plt.tick_params(axis='x', labelsize=35) 
plt.tick_params(axis='y', labelsize=35)

# Scatter plot
scatter = ax.scatter(
    cx, cy,
    color=sns.xkcd_palette(["forest green"])[0],  # Use a valid XKCD color name
    s=60,
    edgecolors="w",
    linewidths=1,
    zorder=2,  # Set a lower z-order for scatter plot to make it appear below contour lines
)

# Contour plot
contour = sns.kdeplot(
    x=cx,
    y=cy,
    ax=ax,
    bw_method='silverman',  
    levels=[0.01, 0.05, 0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],  
    fill=False,  
    zorder=3,  
    linewidths=2,  
    colors=['#AEEEEE', '#87CEEB', '#4682B4', '#4169E1'],  
)

# Scatter plot for individual object
if not df_obj.empty:
    ax.scatter(cx_obj, cy_obj, color="#ff7f0e", marker="*", s=1200, edgecolors="k", zorder=11)

# The fitted lines
x_values = np.linspace(-5.0, 5.0, 100)
ax.plot(x_values, fitted_line_normal(x_values), 'r-', zorder=6, label='Initial fit')
ax.plot(x_values, fitted_line_sigma_clip(x_values), ls='--', color="r", zorder=8, label='Iterative $\sigma$-clipped fit:')

# Add the equation to the legend
intercept_str = f"{b:.2f}" if b >= 0 else f"- {-b:.2f}"
equation_label = f'$y = {a:.2f}x {intercept_str}$'
ax.plot([], [], ' ', label=equation_label)  # Invisible plot to add the equation to the legend

ax.set(xlim=[-0.4, 2.2], ylim=[-0.6, 1.])

# Annotate range
ax.annotate(cmd_args.Ranger, xy=(0.23, 1.),  xycoords='data', size=25, xytext=(-120, -50), 
            textcoords='offset points', bbox=dict(boxstyle="round4,pad=.5", fc="0.9"))

# Representation of the errors
pro_ri = median(ecx)
pro_rj660 = median(ecy)
print("Median", pro_ri)
axis_coordinates_of_representative_error_bar = (0.08, 0.85)
screen_coordinates_of_representative_error_bar = ax.transAxes.transform(axis_coordinates_of_representative_error_bar)
screen_to_data_transform = ax.transData.inverted().transform
data_coordinates_of_representative_error_bar = screen_to_data_transform(screen_coordinates_of_representative_error_bar)
foo = data_coordinates_of_representative_error_bar

ax.errorbar(foo[0], foo[1], xerr=pro_ri, yerr=pro_rj660, c="k", capsize=3)
# ax.annotate("Median Errors", xy=(0.23, 0.89),  xycoords='data', size=25,
#             xytext=(-120, -60), textcoords='offset points', )

if not df_obj.empty: 
    plt.text(cx_obj - 0.1, cy_obj - 0.1, r"Hα emitter", verticalalignment='bottom',
             horizontalalignment='left', fontsize=25, fontweight=None, color='black')

# Add text for the equation of the fitted line
#ax.text(0.65, 0.1, f'$y = {a:.2f}x + {b:.2f}$', transform=ax.transAxes, fontsize=25, verticalalignment='top')

# Update the legend with the equation
ax.legend(loc='lower right', ncol=1, fontsize=25, title='Fitted models', title_fontsize=30)

# Ensure the 'Figs' directory exists
if not os.path.exists('Figs'):
    os.makedirs('Figs')

# Extract file name for saving
save_file = file_.split("-PSF-")[-1].split("_class")[0]

# Construct the full path for saving the file
output_file = os.path.join('Figs', f"color-color-diagram_{save_file}_{cmd_args.Field}.pdf")

# Save the figure
try:
    plt.savefig(output_file, dpi=200)
    print(f"Figure saved successfully as {output_file}")
except Exception as e:
    print(f"Error saving figure: {e}")
finally:
    plt.close(fig)


