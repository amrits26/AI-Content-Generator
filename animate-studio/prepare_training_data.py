# requirements.txt snippet
pandas
tqdm
Pillow

import os
import json
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd

# --- Paths ---
colibri_dir = Path('datasets/colibri')
ot_sien_dir = Path('datasets/ot_sien')
out_dir = Path('datasets/cute_combined')
out_dir.mkdir(parents=True, exist_ok=True)
meta_out = out_dir / 'metadata.jsonl'

# --- Colibri: Filter and Caption ---
colibri_csv = colibri_dir / 'ColibriImagesMetadataAnnotations.csv'
colibri_img_root = colibri_dir
colibri_caption = "vintage children's book illustration, cute animal characters, soft pastel colors, storybook style, gentle expression"

colibri_df = pd.read_csv(colibri_csv)
colibri_df = colibri_df[colibri_df['groundtruth'] == 'Abbildung']

colibri_imgs = []
for _, row in tqdm(colibri_df.iterrows(), total=len(colibri_df), desc='Colibri'):
    img_path = colibri_img_root / row['filename']
    if not img_path.exists():
        continue
    try:
        im = Image.open(img_path)
        if min(im.size) < 256:
            continue
        colibri_imgs.append({'file': img_path, 'caption': colibri_caption})
    except Exception:
        continue

# --- Ot & Sien: Manual/Auto ---
ot_caption = "classic children's book illustration, playful scene, storybook style"
ot_imgs = []
ot_ann_files = list(ot_sien_dir.glob('**/*.json'))
if ot_ann_files:
    for ann_file in ot_ann_files:
        with open(ann_file, 'r') as f:
            anns = json.load(f)
        for ann in anns.get('images', []):
            img_path = ot_sien_dir / ann['file_name']
            if not img_path.exists():
                continue
            try:
                im = Image.open(img_path)
                for obj in ann.get('objects', []):
                    if obj.get('category') in ['animal', 'child', 'children']:
                        bbox = obj['bbox']
                        crop = im.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
                        if min(crop.size) < 256:
                            continue
                        crop_path = out_dir / f"ot_crop_{img_path.stem}_{obj['id']}.jpg"
                        crop.save(crop_path)
                        ot_imgs.append({'file': crop_path, 'caption': ot_caption})
            except Exception:
                continue
else:
    for img_path in tqdm(list(ot_sien_dir.glob('**/*.jpg')), desc='Ot & Sien'):
        try:
            im = Image.open(img_path)
            if min(im.size) < 256:
                continue
            ot_imgs.append({'file': img_path, 'caption': ot_caption})
        except Exception:
            continue

# --- Merge and Limit to 5000 ---
all_imgs = colibri_imgs + ot_imgs
random.shuffle(all_imgs)
all_imgs = all_imgs[:5000]

with open(meta_out, 'w', encoding='utf-8') as fout:
    for entry in tqdm(all_imgs, desc='Writing metadata'):
        rel_path = os.path.relpath(entry['file'], out_dir)
        fout.write(json.dumps({'file_name': rel_path, 'text': entry['caption']}) + '\n')

print(f"✅ Prepared {len(all_imgs)} images in {out_dir}\nMetadata: {meta_out}")
