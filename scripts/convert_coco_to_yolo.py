import json
import os

def load_split_list(split_file):
    with open(split_file, "r") as f:
        return [os.path.splitext(os.path.basename(line.strip()))[0] for line in f if line.strip()]

def convert_coco_to_yolo(coco_json,
                         train_list="data/processed/coco/train.txt",
                         val_list="data/processed/coco/val.txt",
                         labels_root="data/processed/coco/labels"):
    with open(coco_json, "r") as f:
        coco = json.load(f)

    categories = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}
    images = {img["id"]: img for img in coco["images"]}

    train_ids = set(load_split_list(train_list))
    val_ids = set(load_split_list(val_list))

    train_lbl_dir = os.path.join(labels_root, "train")
    val_lbl_dir = os.path.join(labels_root, "val")
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for image_id, anns in anns_by_image.items():
        img = images[image_id]
        file_base = os.path.splitext(img["file_name"])[0]
        width, height = img["width"], img["height"]

        if file_base in train_ids:
            label_path = os.path.join(train_lbl_dir, f"{file_base}.txt")
        elif file_base in val_ids:
            label_path = os.path.join(val_lbl_dir, f"{file_base}.txt")
        else:
            continue

        seen = set()
        with open(label_path, "w") as lf:
            for ann in anns:
                x, y, w, h = ann["bbox"]
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w /= width
                h /= height
                class_id = categories[ann["category_id"]]
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                if line not in seen:
                    lf.write(line + "\n")
                    seen.add(line)

    print(f"✅ Conversion complete. Labels saved in {labels_root}/train and {labels_root}/val")

if __name__ == "__main__":
    convert_coco_to_yolo(
        coco_json="data/processed/coco/annotations_coco.json",
        train_list="data/processed/coco/train.txt",
        val_list="data/processed/coco/val.txt",
        labels_root="data/processed/coco/labels"
    )

