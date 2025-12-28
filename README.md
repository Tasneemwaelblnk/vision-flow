# VisionFlow

VisionFlow is a modular, high-performance Python library designed for building Face Verification datasets.  
It handles the entire pipeline: asynchronous downloading, data cleaning, and restructuring into standard formats (like DocFace/ArcFace) for model training.

## 🚀 Key Features

### ⚡ High-Speed Downloader
Asyncio-based downloader capable of handling 50+ concurrent connections with smart retries and connection pooling.

### 🛠 Modular Architecture
"Plug-and-play" components. Swap out file naming logic, directory structures, or image processors without rewriting core code.

### 📂 Dataset Restructuring
Automatically converts flat CSV data into hierarchical folder structures (e.g., Person_ID/A.jpg, Person_ID/B.jpg) required by training frameworks.

```plaintext
Person_ID/
├── A.jpg
└── B.jpg
```

### 🖼 Image Processing
Integrated tools for cleaning images (e.g., removing black borders from selfies) on the fly.

### 📊 Flexible Data Management
CLI tools to:
- Split Train/Test sets by identity (preventing leakage)
- Filter datasets based on complex logic (Intersection / Union)

## 📦 Installation

To use VisionFlow, install it in editable mode. This allows you to modify the library code and have changes apply instantly.

```bash
cd face_verification_project
pip install -e .
```

Requirements:
- Python 3.8+
- aiohttp
- pandas
- tqdm
- Pillow
- numpy

## 🛠 Modules Overview

### 1. IO (Downloading)
Handles network operations efficiently.

```python
from visionflow.io import AsyncDownloader

# Download a list of (URL, Path) tuples with 50+ concurrent connections
downloader = AsyncDownloader(concurrency=50)
await downloader.download_batch(tasks, description="Downloading IDs")
```

### 2. Ops (Restructuring)
Organizes raw files into training-ready folder structures.

```python
from visionflow.ops import DatasetOrganizer

organizer = DatasetOrganizer("/output/train_set")
organizer.build_id_map(records, group_col="national_id")

# Task A: Create symlinks for ID cards
organizer.process_task(records, "path_id", "national_id", "A", "symlink")

# Task B: Process Selfies with a custom function
def clean_and_save(src, dst):
    img = Image.open(src)
    # Perform operations (resize, crop, etc.)
    img.save(dst)

organizer.process_task(
    records, 
    "path_selfie", 
    "national_id", 
    "B", 
    action="process",           # Use 'process' action
    transform_func=clean_and_save # Pass your function here
)
```

### 3. Core (Data Management)
Wraps Pandas for easy CSV handling and row-level operations.

```python
from visionflow.core import DataManager

dm = DataManager("data.csv").filter(lambda r: r["age"] > 18)
```

## 🏃‍♂️ Quick Start Scripts

The package comes with ready-to-use scripts in the scripts/ directory.

### 1. Download Data
Reads a CSV and downloads images to a local temp folder.

```bash
python scripts/01_download.py
```

(Configuration: Edit TASKS_CONFIG in the script to map your CSV columns to folders)

### 2. Restructure Data (DocFace Format)
Converts downloaded data into Person_ID folders, creating pairs for training.

```bash
python scripts/02_restructure.py
```

Strict Mode:
Only creates folders if both ID and Selfie exist (Best for Training).

Loose Mode:
Creates folders even if one image is missing (Best for Database/Gallery).

### 3. Data Management Tools

Located in scripts/docface_data_managing/.

Split Train/Test:
```bash
python 01_train_test_split.py
```

Extract Unique IDs:
```bash
python 02_extract_unique_ids.py
```

## 📂 Project Structure

```plaintext
visionflow/
├── core/           # DataManager wrapper
├── io/             # AsyncDownloader & Network logic
├── ops/            # DatasetOrganizer (Restructuring logic)
├── processors/     # Image cleaning (Black border removal)
└── utils/          # Logging & Path helpers
```

## 🤝 Contributing

Standardize new logic in visionflow/ops/.

Add new image processors in visionflow/processors/.

Always run pip install -e . if you change folder structures.
