# scripts/preprocess_synthetic_to_yolo.py
import os, csv
from pathlib import Path
from PIL import Image
import random

SRC_CSV = r"data\synthetic\dimensions.csv"  # or synthetic_dataset.csv
IMAGES_DIR = Path(r"data\synthetic\images")
LABELS_DIR = Path(r"data\synthetic\labels")
OUT_YAML = Path(r"data\synthetic\data.yaml")
VAL_SPLIT = 0.2
CLASS_NAMES = ["object"]  # replace if you have multiple classes

def to_yolo_line(cls, x1, y1, x2, y2, w_img, h_img):
    cx = ((x1 + x2) / 2) / w_img
    cy = ((y1 + y2) / 2) / h_img
    w = (x2 - x1) / w_img
    h = (y2 - y1) / h_img
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def main():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_image = {}
    with open(SRC_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_name = r["image"]
            x1, y1, x2, y2 = map(float, (r["x1"], r["y1"], r["x2"], r["y2"]))
            cls = int(r.get("class", 0))
            rows_by_image.setdefault(img_name, []).append((cls, x1, y1, x2, y2))

    images = list(rows_by_image.keys())
    random.shuffle(images)
    n_val = int(len(images) * VAL_SPLIT)
    val_set = set(images[:n_val])

    train_list, val_list = [], []

    for img_name, boxes in rows_by_image.items():
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            print(f"WARNING: missing image {img_name}, skipping")
            continue
        with Image.open(img_path) as im:
            w_img, h_img = im.size

        label_path = LABELS_DIR / (Path(img_name).stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as lf:
            for cls, x1, y1, x2, y2 in boxes:
                lf.write(to_yolo_line(cls, x1, y1, x2, y2, w_img, h_img) + "\n")

        # Collect split lists with full OS paths for data.yaml
        img_full = str(img_path.resolve())
        if img_name in val_set:
            val_list.append(img_full)
        else:
            train_list.append(img_full)

    # Write data.yaml
    data_yaml = f"""# AI_VolumeNet synthetic dataset
path: {IMAGES_DIR.parent.resolve()}
train: {str((IMAGES_DIR.parent / 'train.txt').resolve())}
val: {str((IMAGES_DIR.parent / 'val.txt').resolve())}
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    OUT_YAML.write_text(data_yaml, encoding="utf-8")

    # Write train/val txt files with image paths (Ultralytics supports list files)
    (IMAGES_DIR.parent / "train.txt").write_text("\n".join(train_list), encoding="utf-8")
    (IMAGES_DIR.parent / "val.txt").write_text("\n".join(val_list), encoding="utf-8")
    print("Done: labels created, data.yaml written, train/val split ready.")

if __name__ == "__main__":
    main()
