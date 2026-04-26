"""
Prepare Colibri dataset for LoRA training.
Filters illustrations (groundtruth == 'Abbildung'), copies up to 5000 images,
and creates metadata.jsonl.
"""
import csv
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Paths
DATASET_DIR = Path("datasets")
CSV_PATH = DATASET_DIR / "ColibriImagesMetadataAnnotations.csv"
OUTPUT_DIR = DATASET_DIR / "cute_combined"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGES = 5000
MIN_SIZE = 256

print("📊 Reading metadata CSV...")
good_filenames = set()
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('groundtruth') == 'Abbildung':
            file_path = row.get('file', '')
            if file_path:
                # Extract just the filename from the path
                good_filenames.add(Path(file_path).name)

print(f"✅ Found {len(good_filenames)} illustration filenames in CSV.")

print("🔍 Scanning for image files...")
# Find all images recursively, excluding the output directory itself
image_files = []
for ext in ['*.jpg', '*.jpeg', '*.png']:
    for img_path in DATASET_DIR.rglob(ext):
        if OUTPUT_DIR not in img_path.parents and img_path.parent != OUTPUT_DIR:
            image_files.append(img_path)

print(f"✅ Found {len(image_files)} total image files.")

# Process and copy images
metadata_lines = []
count = 0
skipped_not_in_csv = 0
skipped_small = 0
skipped_error = 0

for img_path in tqdm(image_files, desc="Processing images"):
    if img_path.name not in good_filenames:
        skipped_not_in_csv += 1
        continue
    if count >= MAX_IMAGES:
        break
    try:
        with Image.open(img_path) as img:
            if img.width < MIN_SIZE or img.height < MIN_SIZE:
                skipped_small += 1
                continue
            # Convert to RGB and save as JPEG
            dest_name = f"{count:05d}.jpg"
            dest_path = OUTPUT_DIR / dest_name
            img.convert("RGB").save(dest_path, "JPEG")

        caption = "vintage children's book illustration, cute animal characters, soft pastel colors, storybook style, gentle expression, ultra cute"
        metadata_lines.append({"file_name": dest_name, "text": caption})
        count += 1
    except Exception:
        skipped_error += 1
        continue

# Write metadata.jsonl
metadata_path = OUTPUT_DIR / "metadata.jsonl"
with open(metadata_path, 'w', encoding='utf-8') as f:
    for entry in metadata_lines:
        f.write(json.dumps(entry) + "\n")

print("\n" + "="*50)
print(f"✅ Dataset ready at {OUTPUT_DIR}")
print(f"   Images copied: {count}")
print(f"   Skipped (not in CSV): {skipped_not_in_csv}")
print(f"   Skipped (too small): {skipped_small}")
print(f"   Skipped (errors): {skipped_error}")
print(f"   Metadata file: {metadata_path}")
print("="*50)
