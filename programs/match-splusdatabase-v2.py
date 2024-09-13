import pandas as pd
import splusdata
from astropy.coordinates import SkyCoord
import astropy.units as u
from getpass import getpass

def main():
    # Load your catalog with planetary nebulae
    local_catalog = pd.read_csv("Tab_2_J_A+A_607_A11_table1.csv")
    
    # Convert RA/DEC in the local catalog to SkyCoord objects
    local_coords = SkyCoord(ra=local_catalog['RA'].values * u.deg, 
                            dec=local_catalog['DEC'].values * u.deg, frame='icrs')

    # Your query template for the S-PLUS data
    query_template = """
    SELECT 
    detection.Field, detection.ID, detection.RA, detection.DEC, 
    detection.FWHM, detection.KRON_RADIUS, 
    r.r_PStotal,  r.e_r_PStotal,
    j0660.j0660_pstotal, j0660.e_j0660_pstotal,
    i.i_PStotal,  i.e_i_PStotal
    FROM "idr4_dual"."idr4_detection_image" AS detection 
    LEFT OUTER JOIN "idr4_dual"."idr4_dual_r" AS r ON detection.ID = r.ID
    LEFT OUTER JOIN "idr4_dual"."idr4_dual_j0660" AS j0660 ON j0660.ID = detection.ID
    LEFT OUTER JOIN "idr4_dual"."idr4_dual_i" AS i ON detection.ID = i.ID
    WHERE detection.Field = '{field}'
"""
    
    # Load the S-PLUS field data
    fields = pd.read_csv("https://splus.cloud/files/documentation/iDR4/tabelas/iDR4_zero-points.csv")
    
    # Connect to S-PLUS
    username = input("S-PLUS Username: ")
    password = getpass("S-PLUS Password: ")
    
    try:
        conn = splusdata.connect(username, password)
    except Exception as e:
        print(f"Error connecting to S-PLUS: {e}")
        return
    
    # Create a DataFrame to store the crossmatched results
    crossmatched_table = pd.DataFrame()

    # Iterate over fields in S-PLUS
    for field in fields["Field"]:
        print(f"Querying S-PLUS data for field: {field}")
        query = query_template.format(field=field)
        
        try:
            splus_data = conn.query(query).to_pandas()
        except Exception as e:
            print(f"Error querying field {field}: {e}")
            continue
        
        # Convert S-PLUS RA/DEC to SkyCoord objects
        splus_coords = SkyCoord(ra=splus_data['RA'].values * u.deg, 
                                dec=splus_data['DEC'].values * u.deg, frame='icrs')
        
        # Perform crossmatch using a small separation limit (e.g., 1 arcsec)
        idx, sep2d, _ = splus_coords.match_to_catalog_sky(local_coords)
        sep_constraint = sep2d < 2 * u.arcsec
        matched_splus = splus_data[sep_constraint]
        matched_local = local_catalog.iloc[idx[sep_constraint]]

        # Combine the matched S-PLUS and local data
        matched_table = pd.concat([matched_local.reset_index(drop=True), matched_splus.reset_index(drop=True)], axis=1)

        # Append to the final table
        crossmatched_table = pd.concat([crossmatched_table, matched_table], ignore_index=True)
        
        print(f"Found {len(matched_table)} matches for field {field}")

    # Save the crossmatched results to a CSV file
    crossmatched_table.to_csv("Tab_2_J_A+A_607_A11_table1-splus-filters.csv", index=False)
    print("Crossmatch complete! Results saved to crossmatched_splus_catalog.csv")

if __name__ == "__main__":
    main()
