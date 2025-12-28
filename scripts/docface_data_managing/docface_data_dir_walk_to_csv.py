from pathlib import Path
import csv

dataset_root = Path("/home/tasneem/repos/bassil_face/disjoint_no_shuffle/docface_data_set_final/test_set")  # change if needed
output_csv = "dataset.csv"

rows = []

for identity_dir in dataset_root.iterdir():
    if not identity_dir.is_dir():
        continue

    selfie_path = None
    id_path = None

    for file in identity_dir.iterdir():
        if not file.is_file():
            continue

        name = file.name  #.lower()

        if name.startswith("B_"):
            selfie_path = file
        elif name.startswith("A_"):
            id_path = file

    if selfie_path and id_path:
        rows.append({
            "selfie": str(selfie_path),
            "id": str(id_path)
        })
    else:
        print(f"⚠️ Skipping {identity_dir} (missing selfie or id)")

# write csv
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["selfie", "id"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ CSV created: {output_csv} ({len(rows)} rows)")
