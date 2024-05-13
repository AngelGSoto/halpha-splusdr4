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

# Reddening vector
def redde_vector(x0, y0, x1, y1, a, b, c, d):
    plt.arrow(x0+a, y0+b, (x1+a)-(x0+a), (y1+b)-(y0+b),  fc="k", ec="k", width=0.01,
              head_width=0.07, head_length=0.15) #head_width=0.05, head_length=0.1)
    plt.text(x0+a+c, y0+b+d, 'A$_\mathrm{V}=2$', va='center', fontsize=23)

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


cmd_args = parser.parse_args()
file_ = cmd_args.fileName + ".csv"

datadir = "Disk_Lomeli/"

try:
    df = pd.read_csv(os.path.join(datadir, file_))
except FileNotFoundError:
    df = pd.read_csv(file_)



# Creating the color to creating the diagram from IPHAS
cx, cy = colour(df, "r", "i", "r", "F660")
# error
ecx, ecy = errormag(df, "rerr", "ierr", "rerr", "F660err")

# Modify the file pattern to match the JSON files
patterns = ["../MS_stars/*.json", "MS_stars/*.json"]
file_list = []

for pattern in patterns:
    try:
        file_list = glob.glob(pattern)
        if file_list:
            break  # Exit the loop if files are found
    except FileNotFoundError:
        continue  # Continue to the next pattern if FileNotFoundError occurs

if not file_list:
    print("No JSON files found.")
    sys.exit(1)  # Exit the script if no JSON files are found

A1_MS, B1_MS, A1_giant, B1_giant = plot_mag("F0626_rSDSS", "F0660", "F0769_iSDSS")

# Convert lists to numpy arrays
A1_MS = np.array(A1_MS)
B1_MS = np.array(B1_MS)
A1_giant = np.array(A1_giant)
B1_giant = np.array(B1_giant)

# Plotting
with sns.axes_style("ticks"):
    fig, ax = plt.subplots(figsize=(15, 11))
    #ax.spines["top"].set_visible(False)  
    #ax.spines["right"].set_visible(False)
    plt.xlabel(r"$r - i$", fontsize=35)
    plt.ylabel(r"$r - J0660$", fontsize=35)
    plt.tick_params(axis='x', labelsize=35) 
    plt.tick_params(axis='y', labelsize=35)

    # Create a density plot
    density = ax.hexbin(cx, cy, cmap="nipy_spectral", gridsize=500, mincnt=1.0)

    # Add a colorbar with improved visibility
    #cbar = plt.colorbar(density, ax=ax, orientation='vertical', pad=0.03, format='%.1f')
    #cbar.set_label("Density", fontsize=30)  # Provide a label for the colorbar

    ax.set(
        xlim=[-0.7, 2.6],
        ylim=[-0.8, 1.5])

    # Sort the data points based on B1 (x-axis) for both MS stars and Giants
    MS_sorted_indices = np.argsort(B1_MS)
    giant_sorted_indices = np.argsort(B1_giant)

    # Create lines for MS stars
    ax.plot(B1_MS[MS_sorted_indices], A1_MS[MS_sorted_indices], linestyle='-', alpha=0.7, zorder=2, color='lime', label='Main Sequence')

    # Create lines for giants
    ax.plot(B1_giant[giant_sorted_indices], A1_giant[giant_sorted_indices], linestyle='-', alpha=0.7, zorder=2, color='gold', label='Giants')

    # Scatter plot for MS stars
    ax.scatter(B1_MS, A1_MS, alpha=0.7, zorder=3, color='lime', s=100, label='MS Stars')

    # Scatter plot for giants
    ax.scatter(B1_giant, A1_giant, alpha=0.7, zorder=3, color='gold', s=100, label='Giants')

    ax.annotate(cmd_args.Ranger, xy=(0.08, 1.5),  xycoords='data', size=25,
            xytext=(-120, -50), textcoords='offset points', 
            bbox=dict(boxstyle="round4,pad=.5", fc="0.9"),)

    redde_vector(-1.2314754077697903, 2.147731023789999, -0.8273818571912539, 2.1826566358487645, 3., -2.3, 0.3, -0.07) #E=0.7, this was estimate by comparing the desrending with the redenning model PNe, see Gutiérrez-Soto et al. (2020)

    #representation of the errors
    pro_ri = median(ecx)
    pro_rj660 = median(ecy)
    print("Median", pro_ri)
    axis_coordinates_of_representative_error_bar = (0.08, 0.85)
    screen_coordinates_of_representative_error_bar = ax.transAxes.transform(axis_coordinates_of_representative_error_bar)
    screen_to_data_transform = ax.transData.inverted().transform
    data_coordinates_of_representative_error_bar = screen_to_data_transform(screen_coordinates_of_representative_error_bar)
    foo = data_coordinates_of_representative_error_bar

    ax.errorbar(foo[0], foo[1], xerr=pro_ri, yerr=pro_rj660, c="k", capsize=3)
    ax.annotate("Median Errors", xy=(0.09, 1.35),  xycoords='data', size=25,
            xytext=(-120, -60), textcoords='offset points', )

    # Legend
    #ax.legend(fontsize=20)

save_file = file_.split(".csv")[0]
plt.savefig(f"Figs/color-color-diagram_{save_file}.pdf")
