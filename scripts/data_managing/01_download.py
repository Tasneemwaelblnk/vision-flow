import sys
import os

# Add the parent directory (project root) to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ... now your imports will work
from visionflow.core import DataManager
# ...

import asyncio
import os
from visionflow.core import DataManager
from visionflow.io import AsyncDownloader
from visionflow.utils.common import get_extension_from_url

# --- CONFIG ---
CSV_PATH = "/home/tasneem/repos_0/ASYNC_get_fv_data/testing_lib.csv" # <--- PUT YOUR CSV PATH HERE
OUTPUT_BASE = "/home/tasneem/repos_0/Test_project_vision_flow/downloaded_dataaa"
ID_column = "national_id"
TASKS_CONFIG = [
    {"col": "id_front_img", "folder": "ids", "prefix": "id"},
    {"col": "selfie",       "folder": "selfies", "prefix": "selfie"},
]

async def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: File not found: {CSV_PATH}")
        return

    print(f"Loading {CSV_PATH}...")
    # Load just 100 rows for testing
    dm = DataManager(CSV_PATH).slice(start=0, end=100)
    records = dm.get_records()
    
    # PRINT AVAILABLE COLUMNS
    if records:
        print(f"Found Columns: {list(records[0].keys())}")
    
    downloader = AsyncDownloader(concurrency=50)

    for conf in TASKS_CONFIG:
        col_name = conf['col']
        print(f"\n--- Processing Column: '{col_name}' ---")
        
        # Check if column exists
        if records and col_name not in records[0]:
            print(f"⚠️ WARNING: Column '{col_name}' NOT FOUND in CSV!")
            print(f"   Did you mean one of these? {list(records[0].keys())}")
            continue

        tasks = []
        output_dir = os.path.join(OUTPUT_BASE, conf['folder'])
        
        # Debug counts
        valid_urls = 0
        missing_urls = 0

        for row in records:
            url = row.get(col_name)
            nid = row.get('national_id') # Ensure this column also exists!
            
            if not url or not isinstance(url, str) or not url.startswith('http'):
                missing_urls += 1
                continue
            
            valid_urls += 1
            ext = get_extension_from_url(url)
            filename = f"{conf['prefix']}_{nid}.{ext}"
            save_path = os.path.join(output_dir, filename)
            tasks.append((url, save_path))
            
        print(f"   Found {valid_urls} valid URLs.")
        print(f"   Skipped {missing_urls} empty/invalid rows.")
        
        if valid_urls > 0:
            print(f"   Starting download to: {output_dir}")
            await downloader.download_batch(tasks, description=col_name)
        else:
            print("   ❌ No tasks created for this column.")

if __name__ == "__main__":
    asyncio.run(main())