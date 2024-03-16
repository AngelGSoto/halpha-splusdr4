from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from astropy.modeling import models, fitting
import argparse
import os

def fit_linear_model(x, y):
    """Fit a linear model to the given data."""
    fit = fitting.LinearLSQFitter()
    line_init = models.Linear1D()
    fitted_line = fit(line_init, x, y)
    return fitted_line

def calculate_color(table, f1, f2, f3, f4):
    """Calculate color indices from magnitudes."""
    xcolour = table[f1] - table[f2]
    ycolour = table[f3] - table[f4]
    return xcolour, ycolour

def calculate_error_magnitude(table, ef1, ef2, ef3, ef4):
    """Calculate errors of the colors."""
    excolour = np.sqrt(table[ef1]**2 + table[ef2]**2)
    eycolour = np.sqrt(table[ef3]**2 + table[ef4]**2)
    return excolour, eycolour

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Identify H-alpha emitters from S-PLUS data.")
    parser.add_argument("file", type=str, help="Path to the input CSV file")
    args = parser.parse_args()

    # Read the CSV file
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print("Error: Input file not found.")
        exit(1)

    # Identify unique fields in the catalog
    unique_fields = df['Field'].unique()

    # Initialize empty lists to store tables for each field
    tables_per_field = []
    tables_per_field_oriTable = []

    # Iterate through each unique field
    for field in unique_fields:
        # Select data for the current field
        field_data = df[df['Field'] == field]

        # Creating the color to creating the diagram from IPHAS
        cx, cy = calculate_color(field_data, "R_APER_3", "I_APER_3", "R_APER_3", "F660_APER_3")

        # Using the existing script to fit the data
        fitted_line = fit_linear_model(cx, cy)
        fitted_line_, mask = fitting.FittingWithOutlierRemoval(fit_linear_model, sigma_clip, niter=4, sigma=4.0)(cx, cy)
        filtered_data = np.ma.masked_array(cy, mask=mask)

        # Create DataFrame with the new columns
        colum1 = pd.DataFrame(cx, columns=['r - i'])
        colum2 = pd.DataFrame(cy, columns=['r - J0660'])
        data = pd.concat([field_data["RA"], field_data["DEC"], field_data["R_APER_3"], colum1, colum2], axis=1)

        cy_predic = fitted_line_(data['r - i'])
        sigma_fit = mean_squared_error(cy, cy_predic, squared=False)

        # Applying the criterion
        ecx, ecy = calculate_error_magnitude(field_data, "e_R_APER_3", "e_I_APER_3", "e_R_APER_3", "e_F660_APER_3")
        colum_ri = pd.DataFrame(ecx, columns=['e(r - i)'])
        colum_rh = pd.DataFrame(ecy, columns=['e(r - J0660)'])
        data_final = pd.concat([data, colum_ri, colum_rh], axis=1)

        C = 4.0  # Is the constant
        crite = C * np.sqrt(sigma_fit**2 + data_final["e(r - J0660)"]**2) + cy_predic
        mask = data_final["r - J0660"] >= crite
        data_ha = data_final[mask]

        # Append the resulting table for this field to the list
        tables_per_field.append(data_ha)

        # Apply the mask to the original field_data
        field_data_ha = field_data[mask]
        tables_per_field_oriTable.append(field_data_ha)

    # Concatenate tables_per_field into a single DataFrame
    merged_data_per_field = pd.concat(tables_per_field)

    # Concatenate tables_per_field_oriTable into a single DataFrame
    merged_data_per_field_oriTable = pd.concat(tables_per_field_oriTable)

    # Merge the resulting tables from all fields with original data
    merged_table = pd.concat([merged_data_per_field_oriTable, merged_data_per_field], axis=1)

    # Save the merged table
    csv_file = os.path.splitext(args.file)[0] + "-Halpha.csv"
    merged_table.to_csv(csv_file, index=False)

    print("Script execution completed successfully.")

if __name__ == "__main__":
    main()
