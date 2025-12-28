import pandas as pd
from common import extract_id_from_path

# ==========================================
#               CONFIGURATION
# ==========================================

INPUT_CSV = "data.csv"
OUTPUT_TXT = "valid_ids_list.txt"

# Columns to inspect
TARGET_COLUMNS = ["path_id", "path_selfie"]

# Mode Options:
# "union"        -> Returns IDs found in ANY of the columns
# "intersection" -> Returns IDs found in ALL of the columns (must have both)
# "single"       -> Returns IDs found in the FIRST column only
MODE = "intersection"

# ==========================================
#                  LOGIC
# ==========================================

def get_ids_from_col(df, col_name):
    """Helper: returns a SET of IDs extracted from a column"""
    if col_name not in df.columns:
        print(f"⚠️ Warning: Column '{col_name}' not found in CSV.")
        return set()
    return set(df[col_name].apply(extract_id_from_path).dropna().astype(str))

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    final_ids = set()

    if MODE == 'single':
        target = TARGET_COLUMNS[0]
        print(f"Mode Single: Extracting from '{target}'")
        final_ids = get_ids_from_col(df, target)

    elif MODE == 'union':
        print(f"Mode Union: Extracting from {TARGET_COLUMNS}")
        for col in TARGET_COLUMNS:
            final_ids = final_ids.union(get_ids_from_col(df, col))

    elif MODE == 'intersection':
        print(f"Mode Intersection: Extracting overlap of {TARGET_COLUMNS}")
        # Start with the first column's IDs
        if TARGET_COLUMNS:
            final_ids = get_ids_from_col(df, TARGET_COLUMNS[0])
            # Intersect with the rest
            for col in TARGET_COLUMNS[1:]:
                final_ids = final_ids.intersection(get_ids_from_col(df, col))

    # Save Results
    print(f"Found {len(final_ids)} unique IDs matching criteria.")
    
    with open(OUTPUT_TXT, 'w') as f:
        for nid in sorted(list(final_ids)):
            f.write(f"{nid}\n")
    
    print(f"✅ Saved ID list to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()