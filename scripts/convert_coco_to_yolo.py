import json
import os

def convert_coco_to_yolo(coco_json, images_root="data/processed/coco/images", labels_root="data/processed/coco/labels"):
    with open(coco_json, "r") as f:
        coco = json.load(f)

    # Map category IDs to continuous YOLO class IDs
    categories = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}

    # Build image lookup
    images = {img["id"]: img for img in coco["images"]}

    # Ensure train/val label folders exist
    train_lbl_dir = os.path.join(labels_root, "train")
    val_lbl_dir = os.path.join(labels_root, "val")
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Collect annotations per image
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for image_id, anns in anns_by_image.items():
        img = images[image_id]
        file_name = os.path.splitext(img["file_name"])[0]
        width, height = img["width"], img["height"]

        # Decide split based on image path
        if "train" in img["file_name"]:
            label_path = os.path.join(train_lbl_dir, f"{file_name}.txt")
        elif "val" in img["file_name"]:
            label_path = os.path.join(val_lbl_dir, f"{file_name}.txt")
        else:
            # fallback: dump into root labels folder
            label_path = os.path.join(labels_root, f"{file_name}.txt")

        seen = set()
        with open(label_path, "w") as lf:
            for ann in anns:
                # COCO bbox format: [x_min, y_min, width, height]
                x, y, w, h = ann["bbox"]

                # Convert to YOLO format (normalized)
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w /= width
                h /= height

                class_id = categories[ann["category_id"]]
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

                # Deduplicate
                if line not in seen:
                    lf.write(line + "\n")
                    seen.add(line)

    print(f"✅ Conversion complete. Labels saved in {labels_root}/train and {labels_root}/val")

if __name__ == "__main__":
    convert_coco_to_yolo(
        coco_json="data/processed/coco/annotations_coco.json",
        images_root="data/processed/coco/images",
        labels_root="data/processed/coco/labels"
    )
