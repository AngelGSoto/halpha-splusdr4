'''
This is a simply script to make table with the SDSS format for cross-match.
'''
import pandas as pd
from astropy.table import Table
import numpy as np
import argparse
import sys
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="""Make a table for cross-matching with LAMOST survey data""")
parser.add_argument("source", type=str, default="input_catalog", help="Prefix of the input catalog file (default: 'input_catalog')")
parser.add_argument("--datadir", type=str, default="", help="Path to the directory containing the input catalog file. If not provided, assumes the file is in the same directory as the script.")
cmd_args = parser.parse_args()

# Determine the directory containing the input file
if cmd_args.datadir:
    datadir = cmd_args.datadir
else:
    datadir = os.getcwd()

# Construct input file name
file_ = os.path.join(datadir, cmd_args.source + ".csv")

# Check if the file exists
if not os.path.exists(file_):
    raise FileNotFoundError(f"The file {file_} does not exist.")

# Read the catalog table using Pandas
df = pd.read_csv(file_)

# Rename columns if necessary
if "ALPHA" in df.columns:
    df.rename(columns={"ALPHA": "RA"}, inplace=True)
if "DELTA" in df.columns:
    df.rename(columns={"DELTA": "DEC"}, inplace=True)

# Convert DataFrame to Astropy Table
tab = Table.from_pandas(df)

n = len(tab)

sep = np.linspace(1.0/60., 1.0/60., num=n)

tab["Sep"] = sep

n = int(len(tab) / 500.) + 1

tab_list = [] # list
j = 0 # counter

for i in range(n):
    j += 1  
    tab_list.append(tab[500*i:500*j])

n_list = len(tab_list)

t = [Table() for _ in range(n_list)]
for j in range(n_list):
    t[j]["ra"] = tab_list[j]["RA"]
    t[j]["dec"] = tab_list[j]["DEC"]
    t[j]["sep"] = tab_list[j]["Sep"]

# # Save the file
datadir = "SDSS-spectra/"
for m in range(n_list):
    asciifile = file_.replace(".csv", "-MacthSDSS_{}.dat".format(m))
    t[m].write(os.path.join(datadir, asciifile), format="ascii", delimiter=',', overwrite=True)
