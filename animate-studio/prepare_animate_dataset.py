import os
import shutil
import json
from glob import glob

# Settings
SRC_DIR = r"C:\Users\amrit\OneDrive\Documents\AI-Content-Generator\animate-studio\datasets"
DST_DIR = os.path.join(SRC_DIR, "cute_combined")
os.makedirs(DST_DIR, exist_ok=True)

# Caption template
caption = (
    "vintage children's book illustration, cute animal characters, "
    "soft pastel colors, storybook style, gentle expression, ultra cute"
)

# Find images recursively
img_exts = ('.jpg', '.jpeg', '.png')
all_imgs = []
for ext in img_exts:
    all_imgs.extend(glob(os.path.join(SRC_DIR, '**', f'*{ext}'), recursive=True))

# Limit to 5000 images
all_imgs = all_imgs[:5000]

# Copy and rename images, write metadata
metadata = []
for idx, img_path in enumerate(all_imgs, 1):
    fname = f"{idx:05d}.jpg"
    dst_path = os.path.join(DST_DIR, fname)
    shutil.copyfile(img_path, dst_path)
    metadata.append({"file_name": fname, "text": caption})

# Write metadata.jsonl
with open(os.path.join(DST_DIR, "metadata.jsonl"), "w", encoding="utf-8") as f:
    for entry in metadata:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Prepared {len(all_imgs)} images in {DST_DIR}")
