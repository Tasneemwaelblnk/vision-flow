import pandas as pd
import random
from common import get_id_for_row

# ==========================================
#               CONFIGURATION
# ==========================================

INPUT_CSV = "data.csv"
OUTPUT_CSV = "data_split.csv"

# Which columns contain image paths? (Used to extract the ID)
# The script checks the first one; if empty, checks the second, etc.
IMG_COLUMNS = ["path_id", "path_selfie"] 

# How big should the test set be?
# Use float for percentage (0.2 = 20%)
# Use int for specific number (100 = 100 people)
TEST_SIZE = 0.2

# What should the new column be named?
NEW_COLUMN_NAME = "is_test_set"

# ==========================================
#                  LOGIC
# ==========================================

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # 1. Extract IDs for grouping (Temporary)
    print("Extracting IDs from image paths...")
    df['_temp_id'] = df.apply(lambda row: get_id_for_row(row, IMG_COLUMNS), axis=1)
    
    # Get list of unique people found
    valid_ids = df['_temp_id'].dropna().unique().tolist()
    print(f"Found {len(valid_ids)} unique identities.")

    if len(valid_ids) == 0:
        print("❌ Error: Could not extract any IDs from the specified columns.")
        return

    # 2. Determine Split
    random.shuffle(valid_ids)
    
    if isinstance(TEST_SIZE, float) and TEST_SIZE < 1.0:
        count = int(len(valid_ids) * TEST_SIZE) # Percentage
    else:
        count = int(TEST_SIZE) # Absolute number
        
    test_ids_set = set(valid_ids[:count])
    print(f"Selected {len(test_ids_set)} identities for the Test set.")

    # 3. Create the New Column
    # True if the person's ID is in the test set, False otherwise
    df[NEW_COLUMN_NAME] = df['_temp_id'].apply(lambda x: True if x in test_ids_set else False)
    
    # Cleanup temp column
    df.drop(columns=['_temp_id'], inplace=True)
    
    # 4. Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved to {OUTPUT_CSV}")
    print(f"Column '{NEW_COLUMN_NAME}' distribution:")
    print(df[NEW_COLUMN_NAME].value_counts())

if __name__ == "__main__":
    main()