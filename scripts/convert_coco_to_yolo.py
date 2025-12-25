import json
import os

def convert_coco_to_yolo(coco_json, output_dir):
    with open(coco_json, "r") as f:
        coco = json.load(f)

    # Map category IDs to continuous YOLO class IDs
    categories = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}

    # Build image lookup
    images = {img["id"]: img for img in coco["images"]}

    os.makedirs(output_dir, exist_ok=True)

    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        file_name = os.path.splitext(img["file_name"])[0]
        width, height = img["width"], img["height"]

        # COCO bbox format: [x_min, y_min, width, height]
        x, y, w, h = ann["bbox"]

        # Convert to YOLO format (normalized)
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w /= width
        h /= height

        class_id = categories[ann["category_id"]]

        # Write to .txt file
        label_path = os.path.join(output_dir, f"{file_name}.txt")
        with open(label_path, "a") as lf:
            lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"✅ Conversion complete. Labels saved in {output_dir}")

if __name__ == "__main__":
    convert_coco_to_yolo(
        coco_json="data/processed/coco/annotations_coco.json",
        output_dir="data/processed/coco/labels"
    )
