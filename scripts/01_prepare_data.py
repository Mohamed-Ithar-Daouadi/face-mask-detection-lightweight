# 02_prepare_data.py
# Converts Pascal VOC XML annotations to YOLO format
# and splits dataset into train/val/test

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Config ────────────────────────────────────────
RAW_IMAGES = Path("Face_Mask_Detection_Dataset/images")
RAW_ANNOTS = Path("Face_Mask_Detection_Dataset/annotations")
OUT_DIR    = Path("data/yolo_format")

CLASS_MAP = {
    'with_mask':             0,
    'without_mask':          1,
    'mask_weared_incorrect': 2
}

SEED = 50
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST = remaining 15%

# ── Helper: VOC → YOLO ───────────────────────────
def voc_to_yolo(xml_path, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall('object'):
        cls = obj.find('name').text.strip()
        if cls not in CLASS_MAP:
            print(f"  Warning: unknown class '{cls}' in {xml_path.name}")
            continue
        bb   = obj.find('bndbox')
        xmin = float(bb.find('xmin').text)
        ymin = float(bb.find('ymin').text)
        xmax = float(bb.find('xmax').text)
        ymax = float(bb.find('ymax').text)
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        lines.append(f"{CLASS_MAP[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines

# ── Create output folders ─────────────────────────
for split in ['train', 'val', 'test']:
    (OUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

# ── Gather all images ─────────────────────────────
all_images = sorted(RAW_IMAGES.glob('*.png'))
print(f"Total images found: {len(all_images)}")

random.seed(SEED)
random.shuffle(all_images)

n       = len(all_images)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

splits = {
    'train': all_images[:n_train],
    'val':   all_images[n_train:n_train + n_val],
    'test':  all_images[n_train + n_val:]
}

# ── Process each split ────────────────────────────
import cv2

stats = {'train': 0, 'val': 0, 'test': 0}

for split, images in splits.items():
    print(f"\nProcessing {split} ({len(images)} images)...")
    for img_path in images:
        xml_path = RAW_ANNOTS / (img_path.stem + '.xml')
        if not xml_path.exists():
            print(f"  Warning: no annotation for {img_path.name}, skipping")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: could not read {img_path.name}, skipping")
            continue
        h, w = img.shape[:2]

        # Copy image
        shutil.copy(img_path, OUT_DIR / 'images' / split / img_path.name)

        # Convert and save label
        lines = voc_to_yolo(xml_path, w, h)
        label_path = OUT_DIR / 'labels' / split / (img_path.stem + '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))

        stats[split] += 1

# ── Summary ───────────────────────────────────────
print("\n✅ Done!")
print(f"  Train: {stats['train']} images")
print(f"  Val:   {stats['val']} images")
print(f"  Test:  {stats['test']} images")