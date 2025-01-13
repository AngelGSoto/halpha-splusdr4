import splusdata
import getpass
import pandas as pd
from astropy.table import Table, vstack
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import urllib3
from urllib3.exceptions import MaxRetryError, NewConnectionError

# Connecting with S-PLUS database
username = str(input("Login: "))
password = getpass.getpass("Password: ")

try:
    conn = splusdata.connect(username, password)
except Exception as e:
    print(f"Error connecting to S-PLUS database: {e}")
    exit()

# Loading the input CSV file
file_name = "bess_catalog_converted.csv"
df = pd.read_csv(file_name)
print("Number of objects:", len(df))

# Define the optimized query
Query = """
SELECT detection.Field, detection.ID, detection.RA, detection.DEC, 
       detection.FWHM, detection.KRON_RADIUS, 
       r.r_PStotal, r.e_r_PStotal, 
       J0660.J0660_PStotal, J0660.e_J0660_PStotal, 
       i.i_PStotal, i.e_i_PStotal
FROM TAP_UPLOAD.upload AS tap
LEFT OUTER JOIN idr4_dual.idr4_detection_image AS detection 
    ON (1=CONTAINS( POINT('ICRS', detection.RA, detection.DEC), 
        CIRCLE('ICRS', tap.RA_decimal, tap.DEC_decimal, 0.000277777777778)))
LEFT OUTER JOIN idr4_dual.idr4_dual_r AS r
    ON detection.ID = r.ID
LEFT OUTER JOIN idr4_dual.idr4_dual_J0660 AS J0660
    ON detection.ID = J0660.ID
LEFT OUTER JOIN idr4_dual.idr4_dual_i AS i
    ON detection.ID = i.ID
"""

# Split the data into smaller chunks
chunk_size = 100  # Reduced chunk size for faster processing
n = (len(df) + chunk_size - 1) // chunk_size
print('Number of chunks:', n)

df_chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(n)]

# Define function to query each chunk
def query_chunk(a, chunk):
    start_time = datetime.now()
    print(f"Processing chunk {a}... (Start: {start_time})")
    for attempt in range(3):  # Retry logic, try up to 3 times
        try:
            results = conn.query(Query, chunk)
            if isinstance(results, Table):
                print(f"Processing chunk {a} as Table...")
                return results
            elif isinstance(results, pd.DataFrame):
                print(f"Processing chunk {a} as DataFrame...")
                return Table.from_pandas(results)
            else:
                print(f"Unexpected results type for chunk {a}: {type(results)}")
                return None
        except (MaxRetryError, NewConnectionError, Exception) as e:
            print(f"Error occurred while querying chunk {a} (Attempt {attempt+1}/3): {e}")
            if 'relation "TAP_UPLOAD.upload_' in str(e):
                print(f"Skipping retries for chunk {a} due to table creation error.")
                break  # Break the retry loop if it's a table creation error
            time.sleep(2)  # Wait for 2 seconds before retrying
        finally:
            end_time = datetime.now()
            print(f"Chunk {a} processed (End: {end_time}, Duration: {end_time - start_time})")
    return None

# Use ThreadPoolExecutor to parallelize the queries
merged_table_list = []
with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust number of workers if necessary
    futures = [executor.submit(query_chunk, i, chunk) for i, chunk in enumerate(df_chunks)]
    for future in futures:
        result = future.result()
        if result:
            # Remove rows with empty or invalid RA/DEC values
            valid_rows = result[~result['RA'].mask & ~result['DEC'].mask]
            merged_table_list.append(valid_rows)
        else:
            print(f"Chunk {futures.index(future)} returned no results.")
        time.sleep(1)  # Optional: Add delay to prevent hitting server rate limits

# Merge all results if any
if merged_table_list:
    merged_table = vstack(merged_table_list)  # Combine the results
    # Filter out rows with empty or invalid RA/DEC values
    valid_merged_table = merged_table[~merged_table['RA'].mask & ~merged_table['DEC'].mask]
    print("Number of objects with match:", len(valid_merged_table))

    # Save the result
    output_file = Path(file_name).stem + "-splus-filters-test.csv"
    valid_merged_table.to_pandas().to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
else:
    print("No matching objects found.")
