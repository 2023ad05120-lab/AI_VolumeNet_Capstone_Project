import os, json
import numpy as np
from PIL import Image
from pathlib import Path

# Paths
DET_DIR = "runs/detect/predict/labels"
DEPTH_DIR = "runs/depth/val"
IMG_DIR = "runs/detect/predict"
OUT_DIR = "outputs/aligned/val"
os.makedirs(OUT_DIR, exist_ok=True)

# Load class names from YAML
def load_class_names(yaml_path="configs/coco_synthetic.yaml"):
    import yaml
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return names

# Convert YOLO bbox to pixel coordinates
def yolo_to_pixel_bbox(xc, yc, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    x = int((xc * img_w) - bw / 2)
    y = int((yc * img_h) - bh / 2)
    return x, y, int(bw), int(bh)

# Crop depth region and compute stats
def get_depth_stats(depth_map, x, y, w, h):
    crop = depth_map[y:y+h, x:x+w]
    valid = crop[crop > 0]
    if valid.size == 0:
        return None
    stats = {
        "depth_median": float(np.median(valid)),
        "depth_iqr": float(np.percentile(valid, 75) - np.percentile(valid, 25)),
        "valid_ratio": float(valid.size / (w * h))
    }
    return stats

def find_depth_file(base):
    # Try both naming conventions
    for suffix in ["_depth.png", ".png"]:
        candidate = os.path.join(DEPTH_DIR, base + suffix)
        if os.path.exists(candidate):
            return candidate
    return None

def main():
    class_names = load_class_names()
    label_files = sorted(Path(DET_DIR).glob("*.txt"))
    saved = 0

    for txt_path in label_files:
        base = txt_path.stem
        img_path = os.path.join(IMG_DIR, base + ".jpg")
        depth_path = find_depth_file(base)

        print(f"{base}: img={os.path.exists(img_path)}, depth={os.path.exists(depth_path)}")

        if not os.path.exists(img_path) or not depth_path:
            print(f"⚠️ Skipping {base}: missing image or depth map")
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size
        depth = np.array(Image.open(depth_path)).astype(np.float32) / 255.0

        detections = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) >= 6 else None
                name = class_names[cls_id]
                x, y, bw, bh = yolo_to_pixel_bbox(xc, yc, w, h, img_w, img_h)
                stats = get_depth_stats(depth, x, y, bw, bh)
                det = {
                    "class": name,
                    "bbox_xywh_px": [x, y, bw, bh],
                    "conf": conf,
                    "depth_stats": stats
                }
                detections.append(det)

        out_json = {
            "image": base + ".jpg",
            "detections": detections
        }
        with open(os.path.join(OUT_DIR, base + ".json"), "w") as f:
            json.dump(out_json, f, indent=2)
        saved += 1

    print(f"✅ Alignment complete. Saved {saved} JSON files to {OUT_DIR}")

if __name__ == "__main__":
    main()
