'''
This is a simply script to make table with the SDSS format for cross-match.
'''
from astropy.table import Table, vstack
import numpy as np
import argparse
import sys
import os

parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("source", type=str,
                    default=" teste-program",
                    help="Name of catalog, taken the prefix ")

cmd_args = parser.parse_args()
file_ = cmd_args.source + ".ecsv"

tab = Table.read(file_, format="ascii.ecsv")

n = len(tab)

sep = np.linspace(2.0/60., 2.0/60., num=n)

tab["Sep"] = sep

#n_new = len(table)
#n_ = n_new/900.

# coverting in pandas table
df = (tab.to_pandas())
n = int(len(df) / 500.) + 1

df_ = [] # list
j = 0 # counter
d = {} # empty
for i in range(n):
    j += 1  
    df_.append(df.iloc[500*i:500*j])

n_list = len(df_)

t = [Table() for _ in range(n_list)]
for j in range(n_list):
    t[j]["ra"] = df_[j]["RA"]
    t[j]["dec"] = df_[j]["DEC"]
    t[j]["sep"] = df_[j]["Sep"]

#Save the file
datadir = "SDSS-spectra/"
for m in range(n_list):
    asciifile = file_.replace(".ecsv", "-" + str(m) + ".dat")
    t[m].write(os.path.join(datadir, asciifile), format="ascii", delimiter=',', overwrite=True)

















    
