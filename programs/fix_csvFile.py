import pandas as pd

# Load the CSV file
input_file = 'Tab_2_J_A+A_607_A11_table1-SPLUS_ID.csv'
output_file = 'Tab_2_J_A+A_607_A11_table1-SPLUS_ID_clean.csv'

# Function to check if a row is empty or filled with commas
def is_empty_row(row):
    # Check if the entire row (except the 'Field') contains NaNs or empty strings
    return row.drop(labels='Field').isnull().all() or row['Field'].strip() == ''

# Read the CSV, specifying the appropriate delimiter (comma in this case)
df = pd.read_csv(input_file, delimiter=',')

# Remove rows that are considered empty based on our criteria
df_cleaned = df[~df.apply(is_empty_row, axis=1)]

# Save the cleaned dataframe to a new CSV file
df_cleaned.to_csv(output_file, index=False)

print(f"Cleaned CSV saved to {output_file}")
