import splusdata
import getpass
import pandas as pd
from astropy.table import Table, vstack
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime

# Connecting with S-PLUS database
username = str(input("Login: "))
password = getpass.getpass("Password: ")

try:
    conn = splusdata.connect(username, password)
except Exception as e:
    print(f"Error connecting to S-PLUS database: {e}")
    exit()

# Loading the input CSV file
file_name = "Tab_2_J_A+A_607_A11_table1.csv"
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
        CIRCLE('ICRS', tap.RA, tap.DEC, 0.000277777777778)))
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
    except Exception as e:
        print(f"Error occurred while querying chunk {a}: {e}")
        return None
    finally:
        end_time = datetime.now()
        print(f"Chunk {a} processed (End: {end_time}, Duration: {end_time - start_time})")

# Use ThreadPoolExecutor to parallelize the queries
merged_table_list = []
with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust number of workers if necessary
    futures = [executor.submit(query_chunk, i, chunk) for i, chunk in enumerate(df_chunks)]
    for future in futures:
        result = future.result()
        if result:
            merged_table_list.append(result)
        time.sleep(1)  # Optional: Add delay to prevent hitting server rate limits

# Merge all results if any
if merged_table_list:
    merged_table = vstack(merged_table_list)  # Combine the results
    print("Number of objects with match:", len(merged_table))

    # Save the result
    output_file = Path(file_name).stem + "-splus-filters-test.csv"
    merged_table.to_pandas().to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
else:
    print("No matching objects found.")
