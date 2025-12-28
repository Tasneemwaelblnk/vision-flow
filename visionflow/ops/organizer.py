import os
import json
from collections import defaultdict
from typing import Callable, Optional, List

class DatasetOrganizer:
    def __init__(self, output_root: str):
        self.root = output_root
        self.counters = defaultdict(int)
        self.id_map = {}

    def build_id_map(self, records: List[dict], group_col: str, prefix: str = "person_"):
        unique_ids = sorted(list(set(str(r[group_col]) for r in records if r.get(group_col))))
        self.id_map = {uid: f"{prefix}{i}" for i, uid in enumerate(unique_ids)}

        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, 'identity_map.json'), 'w') as f:
            json.dump(self.id_map, f, indent=4)
        print(f"Identity map built with {len(self.id_map)} identities.")

    def process_task(self, records: List[dict], source_col: str, group_col: str, file_prefix: str, action: str = "symlink", transform_func: Optional[Callable] = None):
        if not self.id_map:
            print("Warning: Identity map not built. Call build_id_map() first.")
            return

        print(f"Processing task: {file_prefix} ({action})...")
        success_count = 0

        for row in records:
            src = row.get(source_col)
            uid = str(row.get(group_col))

            if not src or not os.path.exists(str(src)) or uid not in self.id_map:
                continue

            dest_dir = os.path.join(self.root, self.id_map[uid])
            os.makedirs(dest_dir, exist_ok=True)

            key = (dest_dir, file_prefix)
            idx = self.counters[key]
            filename = f"{file_prefix}_{idx:03d}.jpg"
            dest_path = os.path.join(dest_dir, filename)

            try:
                if action == "symlink":
                    if not os.path.exists(dest_path):
                        os.symlink(os.path.abspath(src), dest_path)
                    success_count += 1
                elif action == "process" and transform_func:
                    if not os.path.exists(dest_path):
                        transform_func(src, dest_path)
                    success_count += 1
                self.counters[key] += 1
            except Exception as e:
                print(f"Error on {src}: {e}")

        print(f"Task {file_prefix} finished Processed {success_count} files.")