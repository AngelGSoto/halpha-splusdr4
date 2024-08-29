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

def plot_mag(f1, f2, f3, field_name):
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

parser.add_argument("--Object", type=str,
                    default="122465",
                    help="Choosing an individual object in the Field")

parser.add_argument("--Ranger", type=str,
                    default="r < 16",
                    help="Range on r-magnitude")

cmd_args = parser.parse_args()
file_ = cmd_args.fileName + ".csv"

datadir = "Disk_Lomeli/"

try:
    df = pd.read_csv(os.path.join(datadir, file_))
except FileNotFoundError:
    df = pd.read_csv(file_)

# List of unique fields
fields = df["field"].unique()

# Modify the file pattern to match the JSON files
pattern = "../MS_stars/*.json"
file_list = glob.glob(pattern)

# Iterate through each field and generate plots
for field in fields:
    # Choose the field
    mask_field = df["field"] == field
    df_field = df[mask_field]

    # Choose the object if provided
    mask_obj = df_field["NUMBER"] == cmd_args.Object
    df_obj = df_field[mask_obj]

    # Remove rows corresponding to the individual object from df_field
    df_field = df_field[~mask_obj]

    # Creating the color to creating the diagram from IPHAS
    cx, cy = colour(df_field, "r", "i", "r", "F660")
    # error
    ecx, ecy = errormag(df_field, "rerr", "ierr", "rerr", "F660err")

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
    a = fitted_line_sigma_clip.slope.value
    b = fitted_line_sigma_clip.intercept.value
    print("Field:", field)
    print("Pendiente:", a)
    print("Intercept:", b)

    # Check if any rows are selected
    if df_obj.empty:
        print(f"No data found for object {cmd_args.Object} in field {field}. Skipping individual object plot.")
    else:
        # Creating the color to creating the diagram from IPHAS
        cx_obj, cy_obj = colour(df_obj, "r", "i", "r", "F660")

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
        s=50,
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
    ax.plot(x_values, fitted_line_normal(x_values), 'r-', zorder=6, label='Initial fitted')
    ax.plot(x_values, fitted_line_sigma_clip(x_values), ls='--', color="r", zorder=8, label='Iter. fitted $\\sigma$ clipped:')

    # Add the equation to the legend
    intercept_str = f"{b:.2f}" if b >= 0 else f"- {-b:.2f}"
    equation_label = f'$y = {a:.2f}x {intercept_str}$'
    ax.plot([], [], ' ', label=equation_label)  # Invisible plot to add the equation to the legend

    ax.set(xlim=[-0.4, 1.7], ylim=[-0.3, 0.6])

    # Annotate range
    ax.annotate(cmd_args.Ranger, xy=(0.08, 1.5),  xycoords='data', size=25, xytext=(-120, -50), 
                textcoords='offset points', bbox=dict(boxstyle="round4,pad=.5", fc="0.9"))

    # Representation of the errors
    pro_ri, pro_i_j = [], []
    for xi, yi, e_xi, e_yi in zip(cx, cy, ecx, ecy):
        if e_xi < 0.02 and e_yi < 0.05:
            pro_ri.append(xi)
            pro_i_j.append(yi)
    # Plot the selection region as a rectangle
    # rect = plt.Rectangle((min(pro_ri), min(pro_i_j)), width=(max(pro_ri) - min(pro_ri)),
    #                      height=(max(pro_i_j) - min(pro_i_j)), linewidth=2, edgecolor='green', facecolor='none')
    #ax.add_patch(rect)

    # Save the plot
    out_file_name = f"Fig_color_diagram_disk_16R175/color_color_diagram_{field}.pdf"
    plt.savefig(out_file_name, bbox_inches='tight', dpi=300)
    print(f"Plot saved as {out_file_name}")

    # Close the plot to avoid overlap
    plt.close()











    
