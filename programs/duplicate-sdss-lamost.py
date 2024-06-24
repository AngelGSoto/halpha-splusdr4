from __future__ import print_function
import pandas as pd
import argparse
from sklearn.neighbors import KDTree
from astropy.table import Table

# Parse command-line arguments
parser = argparse.ArgumentParser(description="""Make a table from the S-PLUS catalogs """)
parser.add_argument("fileName", type=str, default="teste-program", help="Name of table, taken the prefix ")
cmd_args = parser.parse_args()
file_ = cmd_args.fileName + ".dat"

tab = Table.read(file_, format='ascii')

# Convert into a pandas DataFrame
data = tab.to_pandas()
print("Number of stars:", len(data))

# Define a threshold for considering coordinates as identical
coordinate_threshold = 0.0001  # Example threshold in degrees

# Create a KD-tree for efficient nearest neighbor search
tree = KDTree(data[['RA', 'DEC']].values, leaf_size=30)

# Identify potential duplicate objects based on the coordinate threshold
potential_duplicates = []
processed_pairs = set()  # To store processed pairs and avoid duplication
for i, row in data.iterrows():
    indices = tree.query_radius([[row['RA'], row['DEC']]], r=coordinate_threshold)
    indices = indices[0]  # Extract the indices from the returned array
    for j in indices:
        if i != j:  # Exclude the object itself
            pair = tuple(sorted([i, j]))  # Sort the indices to handle both orders of pairs
            if pair not in processed_pairs:
                obj1 = data.iloc[i]
                obj2 = data.iloc[j]
                potential_duplicates.append((obj1, obj2))
                processed_pairs.add(pair)

# Print pairs of duplicate objects
print("Pairs of duplicate objects based on similar coordinates:")
for obj1, obj2 in potential_duplicates:
    print("Duplicate Pair:")
    print(obj1[['FileName', 'ID', 'RA', 'DEC']])
    print(obj2[['FileName', 'ID', 'RA', 'DEC']])
    print("-----------------------------")


# Create a set to store indices of duplicate objects to remove
duplicate_indices_to_remove = set()
for obj1, obj2 in potential_duplicates:
    # Keep the index of obj2 to remove it later
    duplicate_indices_to_remove.add(obj2.name)

# Remove duplicate objects and keep one representative object
unique_objects = data.drop(index=duplicate_indices_to_remove)

# Save the new table without duplicate objects to a CSV file
unique_objects.to_csv("spectra-information-lamost-unique-unique.csv", index=False)
