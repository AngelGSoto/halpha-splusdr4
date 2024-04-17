'''
This is a simply script to make table with the SDSS format for cross-match.
'''
import pandas as pd
from astropy.table import Table
import numpy as np
import argparse
import sys
import os

parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("source", type=str,
                    default="teste-program",
                    help="Name of catalog, taken the prefix ")

cmd_args = parser.parse_args()
file_ = cmd_args.source + ".ecsv"

# Check if the file is a CSV
if os.path.splitext(file_)[1] == ".csv":
    file_format = "pandas_csv"
else:
    file_format = "ascii.ecsv"

# Read the file based on the format
if file_format == "pandas_csv":
    df = pd.read_csv(file_)
    tab = Table.from_pandas(df)
else:
    tab = Table.read(file_, format="ascii.ecsv")

n = len(tab)

sep = np.linspace(2.0/60., 2.0/60., num=n)

tab["Sep"] = sep

n = int(len(tab) / 500.) + 1

tab_list = [] # list
j = 0 # counter

for i in range(n):
    j += 1  
    tab_list.append(tab[500*i:500*j])

n_list = len(tab_list)

# Save the file
datadir = "SDSS-spectra/"
for m in range(n_list):
    asciifile = file_.replace(".ecsv", "-{}.dat".format(m))
    tab_list[m].write(os.path.join(datadir, asciifile), format="ascii", delimiter=',', overwrite=True)
