import splusdata
import getpass
import pandas as pd
from astropy.table import Table, vstack
import numpy as np
import argparse
from pathlib import Path
  
ROOT = Path("Final-list/")
  
# Connecting with SPLUS database
username = str(input("Login: "))
password = getpass.getpass("Password: ")

conn = splusdata.connect(username, password)
  
parser = argparse.ArgumentParser(
    description="""Get data form splus.cloudy""")

parser.add_argument("rmagi", type=float,
                    default="12",
                    help="r-mag; initial")

parser.add_argument("rmagf", type=float,
                    default="16",
                    help="r-mag; final")

cmd_args = parser.parse_args()
ri = cmd_args.rmagi
rf = cmd_args.rmagf

rii = np.linspace(ri, rf, 10)

riii = []
rfff = []
for i, j in enumerate(rii, start = 1):
    if j < rf:
         riii.append(j)
    if j > ri:
        rfff.append(j)

merged_table_list = []
for k, t in zip(riii, rfff):
    Query = f"""SELECT r.Field, r.ID, r.RA_r, r.DEC_r, r.X_r, r.Y_r, r.s2n_r_psf, j0660.s2n_J0660_psf,
                  i.s2n_i_psf, r.CLASS_STAR_r, i.CLASS_STAR_i, 
                  r.r_psf, j0660.J0660_psf, i.i_psf, r.e_r_psf, j0660.e_J0660_psf, i.e_i_psf 
        	  FROM "idr4_psf"."idr4_psf_r" as r JOIN "idr4_psf"."idr4_psf_j0660" as j0660 ON r.ID=j0660.ID 
                  JOIN "idr4_psf"."idr4_psf_i" as i ON r.ID=i.ID 
                  WHERE CLASS_STAR_r > 0.6 AND CLASS_STAR_i > 0.6  
                  AND r_psf >= """ + str(k) + """ AND r_psf < """  + str(t)  +  """
                  AND s2n_r_psf > 5 AND s2n_J0660_psf > 5 AND s2n_i_psf > 5
                  AND e_r_psf <= 0.2 AND e_J0660_psf <= 0.2 AND e_i_psf <= 0.2
                  AND X_r >= 1400 AND X_r <= 10100 AND Y_r >= 1200 AND Y_r <= 9500"""
    # Executing the query
    results = conn.query(Query)
    print("Number of soures in the interval:", "[", str(k), ",", str(t), "]", "=>", len(results))
    merged_table_list.append(results)
    

# Merging all result astropy tables 
merged_table = vstack(merged_table_list)

# coverting in pandas table
df = (merged_table.to_pandas())

df.to_csv("iDR4-SPLUS-psf-STAR-{}r{}-StN5-err02.csv".format(int(ri), int(rf)), index = False)
