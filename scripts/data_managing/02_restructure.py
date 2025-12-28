import sys
import os
import re
import pandas as pd
from PIL import Image
# cd /home/tasneem/repos_0/Test_project_vision_flow/face_verification_project
# pip install -e .
# --- PATH SETUP (To find visionflow package) ---
# This allows imports to work without 'pip install -e .' (though installation is recommended)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- PATH SETUP ---
# This goes up TWO levels (to 'face_verification_project/') - CORRECT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from visionflow.core import DataManager
from visionflow.ops import DatasetOrganizer
from visionflow.processors import remove_black_border

# ==========================================
#               CONFIGURATION
# ==========================================

# 1. Path to your matched CSV
CSV_PATH = "/home/tasneem/repos_0/ASYNC_get_fv_data/face_verification_train_dataset.csv"

# 2. Output directory
OUTPUT_DIR = "/home/tasneem/repos_0/face_verification_version_2/train_set"
# 'frontalized_id', 'frontalized_selfie'
ID_COLUMN_NAME="frontalized_id"
SELFIE_COLUMN_NAME="frontalized_selfie"

# 3. FILTER FLAG
# True  = Skip folder if ID or Selfie is missing (Best for Training)
# False = Keep folder even if only one image exists (Best for Storage)
REQUIRE_BOTH_FILES = True 

# ==========================================
#             HELPER FUNCTIONS
# ==========================================

def extract_national_id_from_path(path):
    """
    Extracts the first sequence of digits found in the filename.
    Ex: /path/to/id_2950101.jpg -> '2950101'
    """
    if not path or pd.isna(path):
        return None
    
    filename = os.path.basename(str(path))
    # Regex: Look for a sequence of digits (\d+) inside the filename
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    return None

def check_both_exist(has_id: bool, has_selfie: bool, strict_flag: bool) -> bool:
    """
    Decides whether to keep the record based on file existence and the flag.
    """
    if strict_flag:
        return has_id and has_selfie # Strict
    else:
        return has_id or has_selfie  # Loose

def clean_and_save(src, dst):
    """Processor to remove black borders and save."""
    try:
        with Image.open(src) as img:
            res = remove_black_border(img)
            res.save(dst, quality=95)
    except Exception as e:
        print(f"Error processing image {src}: {e}")

# ==========================================
#             MAIN EXECUTION
# ==========================================

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: File not found: {CSV_PATH}")
        return

    print(f"Loading Data from: {CSV_PATH}")
    print(f"Strict Mode (Require Both Files): {REQUIRE_BOTH_FILES}")
    
    # 1. Load Data
    dm = DataManager(CSV_PATH)
    records = dm.get_records()

    print(f"Loaded {len(records)} rows. Validating files and IDs...")

    # 2. Pre-process Records
    valid_records = []
    
    for r in records:
        path_id = r.get(ID_COLUMN_NAME)
        path_selfie = r.get(SELFIE_COLUMN_NAME)
        
        # --- A. CHECK PHYSICAL FILES ---
        has_id = path_id and os.path.exists(str(path_id))
        has_selfie = path_selfie and os.path.exists(str(path_selfie))

        # Apply the filter flag (check_both_exist)
        if not check_both_exist(has_id, has_selfie, REQUIRE_BOTH_FILES):
            continue

        # --- B. RESOLVE NATIONAL ID (The Priority Logic) ---
        nid = r.get('national_id')

        # 1. If column is missing or empty, try extracting from ID path
        if pd.isna(nid) or str(nid).strip() == "":
            if has_id:
                nid = extract_national_id_from_path(path_id)
        
        # 2. If still missing, try extracting from Selfie path
        if pd.isna(nid) or str(nid).strip() == "":
            if has_selfie:
                nid = extract_national_id_from_path(path_selfie)

        # 3. If STILL missing, we can't process this person
        if not nid:
            # print(f"Skipping row: Could not find National ID in column or filenames.")
            continue

        # --- C. UPDATE RECORD ---
        r['national_id'] = str(nid) # Ensure it is a string
        r['path_id'] = path_id if has_id else None
        r['path_selfie'] = path_selfie if has_selfie else None
        
        valid_records.append(r)

    print(f"Found {len(valid_records)} valid records ready for restructuring.")

    # 3. Setup Organizer
    organizer = DatasetOrganizer(OUTPUT_DIR)

    # 4. Build Identity Map
    organizer.build_id_map(valid_records, group_col="national_id")

    # 5. Execute Tasks
    
    # Task A: ID Cards (Symlink)
    organizer.process_task(
        records=valid_records, 
        source_col="path_id", 
        group_col="national_id", 
        file_prefix="A", 
        action="symlink"
    )

    # Task B: Selfies (Process & Save)
    organizer.process_task(
        records=valid_records, 
        source_col="path_selfie", 
        group_col="national_id", 
        file_prefix="B", 
        action="symlink",
        # action="process", 
        # transform_func=clean_and_save
    )

if __name__ == "__main__":
    main()