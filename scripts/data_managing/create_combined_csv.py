import os
import pandas as pd
import re

# --- Configuration Section ---

# 1. Define the image subfolders as ABSOLUTE PATHS
# ⚠️ YOU MUST REPLACE THESE PLACEHOLDERS WITH YOUR ACTUAL FULL PATHS!
RAW_ID_FOLDERS = ['/temp/ids', '/temp/face_verification_v2_dataset/ids__2']
RAW_SELFIE_FOLDERS = ['/temp/faces_combined', '/temp/face_verification_v2_dataset/selfies__2']
CROPPED_ID_FOLDERS = ['/temp/cropped_id__1', '/temp/face_verification_v2_dataset/cropped_id__2']
CROPPED_SELFIE_FOLDERS = ['/temp/cropped_selfie__1', '/temp/face_verification_v2_dataset/cropped_selfie__2']
FRONTALIZED_ID_FOLDERS = ['/temp/frontalized_id__1', '/temp/face_verification_v2_dataset/frontalized_id__2']
FRONTALIZED_SELFIE_FOLDERS = ['/temp/frontalized_selfie__1', '/temp/face_verification_v2_dataset/frontalized_selfie__2']

# 2. Define the path to the list of IDs for the 'in_test' flag
TEST_ID_FILE = '/home/tasneem/repos_0/ASYNC_get_fv_data/unique_national_ids.txt' 

# 3. Define the output CSV file name
OUTPUT_CSV = '/home/tasneem/repos_0/ASYNC_get_fv_data/aggregated_master_dataset_with_drop_flag_____22222.csv'

# Define the image type columns for easy iteration and checking completeness
IMAGE_COLUMNS = [
    'raw_id', 'raw_selfie', 'cropped_id', 'cropped_selfie', 
    'frontalized_id', 'frontalized_selfie'
]
# Define the mapping between column names and their folder lists
COLUMN_TO_FOLDERS = {
    'raw_id': RAW_ID_FOLDERS,
    'raw_selfie': RAW_SELFIE_FOLDERS,
    'cropped_id': CROPPED_ID_FOLDERS,
    'cropped_selfie': CROPPED_SELFIE_FOLDERS,
    'frontalized_id': FRONTALIZED_ID_FOLDERS,
    'frontalized_selfie': FRONTALIZED_SELFIE_FOLDERS
}


# --- Utility Functions ---

def load_test_ids(file_path):
    """Loads national IDs from the unique_national_ids.txt file into a set."""
    if not os.path.exists(file_path):
        print(f"Warning: Test ID file not found at {file_path}. 'in_test' column will be False for all.")
        return set()
    try:
        with open(file_path, 'r') as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        print(f"Error reading test ID file: {e}")
        return set()

# MODIFIED: Removed 'base_dir' parameter
def get_image_map(folders):
    """
    Scans specified folders (which are now full paths), extracts the national ID, 
    and creates a dictionary mapping National ID -> (Full Path, Source Folder Name).
    """
    id_map = {}
    id_pattern = re.compile(r'image_(\d{14})')
    
    for full_path in folders: # Iterate over the full paths directly
        # Use os.path.basename() to get a short identifier for the 'folder' column
        folder_name = os.path.basename(full_path) 
        
        if not os.path.isdir(full_path):
            print(f"Warning: Folder not found: {full_path}. Skipping.")
            continue
            
        print(f"Scanning {full_path}...")
        for filename in os.listdir(full_path):
            match = id_pattern.search(filename)
            if match:
                national_id = match.group(1)
                # Store the data: (Full Path, Source Folder Name)
                # Note: We use the full path variable directly as the directory path
                id_map[national_id] = (os.path.join(full_path, filename), folder_name)
            
    return id_map

# --- Main Aggregation Logic ---

def create_master_dataset():
    
    # 1. Load the test IDs
    test_id_set = load_test_ids(TEST_ID_FILE)

    # 2. Get maps for all image types (ID -> (Path, Folder))
    print("\n--- Generating Image Maps ---")
    # MODIFIED: Call get_image_map without base_dir
    map_data = {
        col: get_image_map(folders) 
        for col, folders in COLUMN_TO_FOLDERS.items()
    }

    # 3. Identify all unique national IDs found across ALL image types
    all_national_ids = set()
    for map_ in map_data.values():
        all_national_ids.update(map_.keys())
        
    print(f"\n--- Aggregation ---")
    print(f"Total unique national IDs found across all folders: {len(all_national_ids)}")

    # 4. Build the final dataset row by row
    data_list = []
    
    for national_id in sorted(list(all_national_ids)):
        
        row = {'nationalid': national_id}
        
        # Initialize flags and folder name tracker
        folder_names = set()
        
        # New flag: Assume we don't need to drop it unless a path is missing
        needs_dropping = False 
        
        # Populate the path columns and check for missing files
        for col in IMAGE_COLUMNS:
            map_ = map_data[col]
            path_info = map_.get(national_id)
            
            if path_info:
                path, folder = path_info
                row[col] = path
                folder_names.add(folder)
            else:
                # Path is missing (None)
                row[col] = None 
                needs_dropping = True # 🚨 Set flag to True if any path is missing
        
        # Determine the source folder (simple aggregation for now)
        row['folder'] = ', '.join(sorted(list(folder_names)))
        
        # Set the 'in_test' flag
        row['in_test'] = national_id in test_id_set
        
        # Set the new 'to_drop' flag based on the check above
        row['to_drop'] = needs_dropping 
        
        data_list.append(row)

    # 5. Create DataFrame with the exact desired column order
    df = pd.DataFrame(data_list)
    
    # Define the final column order
    final_cols = IMAGE_COLUMNS + ['folder', 'nationalid', 'in_test', 'to_drop']
    
    # Reorder columns
    df = df.reindex(columns=final_cols)
    
    # 6. Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Master dataset successfully created and saved to **{OUTPUT_CSV}**")
    print(f"Generated {len(df)} records.")
    
    drop_count = df['to_drop'].sum()
    print(f"🚩 Found {drop_count} records marked with 'to_drop' = True (missing one or more required images).")
    print("\n--- Example Rows (first 5) ---")
    print(df.head())


# Execute the function
create_master_dataset()