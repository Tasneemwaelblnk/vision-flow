import pandas as pd
import json
from common import extract_id_from_path

# ==========================================
#               CONFIGURATION
# ==========================================

INPUT_CSV = "data.csv"
IDENTITY_MAP_JSON = "identity_map.json"
OUTPUT_CSV = "data_checked.csv"

# Which column has the image path to check?
IMG_COLUMN_TO_CHECK = "path_id"

# Name of the new column to add
NEW_COLUMN_NAME = "exists_in_map"

# ==========================================
#                  LOGIC
# ==========================================

def main():
    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Loading Map: {IDENTITY_MAP_JSON}")
    try:
        with open(IDENTITY_MAP_JSON, 'r') as f:
            identity_map = json.load(f)
    except FileNotFoundError:
        print("❌ Error: Identity Map JSON file not found.")
        return

    # Create a set of valid keys for fast lookup
    valid_map_keys = set(str(k) for k in identity_map.keys())

    print(f"Checking IDs extracted from '{IMG_COLUMN_TO_CHECK}'...")

    def check_row(path_val):
        # 1. Extract ID from the path
        nid = extract_id_from_path(path_val)
        
        # 2. Check if it exists in the map keys
        if nid and str(nid) in valid_map_keys:
            return True
        return False

    # Apply logic to create new column
    if IMG_COLUMN_TO_CHECK in df.columns:
        df[NEW_COLUMN_NAME] = df[IMG_COLUMN_TO_CHECK].apply(check_row)
    else:
        print(f"❌ Error: Column '{IMG_COLUMN_TO_CHECK}' not found in CSV.")
        return

    # Save
    count_found = df[NEW_COLUMN_NAME].sum()
    print(f"Found {count_found} rows present in the identity map.")
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()