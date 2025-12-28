import os
import re
import pandas as pd

def extract_id_from_path(path_str):
    """
    Scans a file path string and extracts the first sequence of digits.
    Example: /data/selfie_12345.jpg -> '12345'
    """
    if pd.isna(path_str) or str(path_str).strip() == "":
        return None
    
    filename = os.path.basename(str(path_str))
    # Regex: Find the first sequence of digits in the filename
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    return None

def get_id_for_row(row, columns_to_check):
    """
    Tries to find an ID in the specified columns in order.
    """
    for col in columns_to_check:
        if col not in row: continue
        
        val = row[col]
        
        # If the value looks like a file path, try extracting ID
        if isinstance(val, str):
            extracted = extract_id_from_path(val)
            if extracted: return extracted
            
        # If the value is just a number/string, use it directly
        elif pd.notna(val):
            return str(val)
            
    return None