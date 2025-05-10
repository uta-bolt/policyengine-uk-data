# Create a file named fix_dwp_frs.py
from pathlib import Path
import pandas as pd
import warnings
from policyengine_uk_data.datasets.frs.dwp_frs import DWP_FRS, DWP_FRS_2020_21, DWP_FRS_2022_23

def fixed_generate(self):
    """Fixed version of the generate method"""
    tab_folder = self.folder

    if isinstance(tab_folder, str):
        tab_folder = Path(tab_folder)

    tab_folder = Path(tab_folder.parent / tab_folder.stem)
    print(f"Looking for files in: {tab_folder}")
    
    # Load the data
    tables = {}
    for tab_file in tab_folder.glob("*.tab"):
        table_name = tab_file.stem
        if "frs" in table_name:
            continue
            
        print(f"Processing {table_name}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                tables[table_name] = pd.read_csv(
                    tab_file, delimiter="\t"
                ).apply(pd.to_numeric, errors="coerce")
                tables[table_name].columns = tables[table_name].columns.str.upper()
                print(f"Successfully loaded {table_name} with shape {tables[table_name].shape}")
            except Exception as e:
                print(f"Error loading {table_name}: {e}")
                continue

        sernum = (
            "sernum"
            if "sernum" in tables[table_name].columns
            else "SERNUM"
        )  # FRS inconsistently users sernum/SERNUM in different years

        # Only add person_id if all required columns exist
        if all(col in tables[table_name].columns for col in ["PERSON", "BENUNIT", sernum]):
            tables[table_name]["person_id"] = (
                tables[table_name][sernum] * 1e2
                + tables[table_name].BENUNIT * 1e1
                + tables[table_name].PERSON
            ).astype(int)
            print(f"Added person_id to {table_name}")

        # Only add benunit_id if all required columns exist
        if all(col in tables[table_name].columns for col in ["BENUNIT", sernum]):
            tables[table_name]["benunit_id"] = (
                tables[table_name][sernum] * 1e2
                + tables[table_name].BENUNIT * 1e1
            ).astype(int)
            print(f"Added benunit_id to {table_name}")

        # Only add household_id if sernum exists
        if sernum in tables[table_name].columns:
            tables[table_name]["household_id"] = (
                tables[table_name][sernum] * 1e2
            ).astype(int)
            print(f"Added household_id to {table_name}")
            
        # Set indices only if the ID columns exist
        if table_name in ("adult", "child") and "person_id" in tables[table_name].columns:
            tables[table_name].set_index("person_id", inplace=True, drop=False)
            print(f"Set person_id as index for {table_name}")
        elif table_name == "benunit" and "benunit_id" in tables[table_name].columns:
            tables[table_name].set_index("benunit_id", inplace=True, drop=False)
            print(f"Set benunit_id as index for {table_name}")
        elif table_name == "househol" and "household_id" in tables[table_name].columns:
            tables[table_name].set_index("household_id", inplace=True, drop=False)
            print(f"Set household_id as index for {table_name}")
    
    print(f"Tables loaded: {list(tables.keys())}")
    
    # Check if we have all required tables
    required_tables = ["adult", "benunit", "househol"]
    missing_tables = [table for table in required_tables if table not in tables]
    if missing_tables:
        print(f"WARNING: Missing required tables: {missing_tables}")
        
        # If benunit is missing, we need to create it from adult data
        if "benunit" in missing_tables and "adult" in tables:
            print("Creating synthetic benunit table from adult data")
            benunit_ids = tables["adult"]["benunit_id"].unique()
            tables["benunit"] = pd.DataFrame({"benunit_id": benunit_ids})
            tables["benunit"].set_index("benunit_id", inplace=True, drop=False)
    
    # Only filter if both tables exist and have the required columns
    if all(table in tables for table in ["benunit", "adult"]):
        if "benunit_id" in tables["benunit"].columns and "benunit_id" in tables["adult"].columns:
            print("Filtering benunit table...")
            benunit_ids_in_adult = set(tables["adult"]["benunit_id"])
            benunit_ids_in_benunit = set(tables["benunit"]["benunit_id"])
            print(f"Number of benunit IDs in adult table: {len(benunit_ids_in_adult)}")
            print(f"Number of benunit IDs in benunit table: {len(benunit_ids_in_benunit)}")
            print(f"Number of benunit IDs in both: {len(benunit_ids_in_adult & benunit_ids_in_benunit)}")
            
            try:
                tables["benunit"] = tables["benunit"][
                    tables["benunit"].benunit_id.isin(tables["adult"].benunit_id)
                ]
                print(f"Filtered benunit table: {tables['benunit'].shape}")
            except Exception as e:
                print(f"Error filtering benunit table: {e}")
        else:
            print("Cannot filter benunit table - missing benunit_id in one or both tables")
    
    if all(table in tables for table in ["househol", "adult"]):
        if "household_id" in tables["househol"].columns and "household_id" in tables["adult"].columns:
            print("Filtering househol table...")
            try:
                tables["househol"] = tables["househol"][
                    tables["househol"].household_id.isin(tables["adult"].household_id)
                ]
                print(f"Filtered househol table: {tables['househol'].shape}")
            except Exception as e:
                print(f"Error filtering househol table: {e}")
        else:
            print("Cannot filter househol table - missing household_id in one or both tables")

    # Save the data
    print("Saving dataset...")
    self.save_dataset(tables)
    print("Dataset saved successfully")
    return tables  # Return tables for inspection

# Monkey patch the generate method
DWP_FRS.generate = fixed_generate

# Run the generate method
print("Running generate method with fixed implementation")
instance = DWP_FRS_2020_21()
tables = instance.generate()
print("Generate method completed")

# Optionally, run for 2022-23 as well
print("\nNow running for 2022-23...")
instance2 = DWP_FRS_2022_23()
tables2 = instance2.generate()
print("2022-23 generate method completed")