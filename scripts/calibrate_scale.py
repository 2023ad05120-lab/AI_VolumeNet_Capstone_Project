import os, glob, json, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import yaml

# Reference anchors with known real-world sizes
REFERENCE_SIZES_MM = {
    "plate": {"type": "circle", "diameter_mm": 240.0},   # dinner plate
    "pan": {"type": "circle", "diameter_mm": 280.0},     # frying pan
    "mug": {"type": "cylinder", "diameter_mm": 80.0, "height_mm": 95.0},  # coffee mug
}

# Other classes to scale and compute dimensions/volumes
TARGET_CLASSES = ["cup", "bottle", "bowl", "fork", "spoon", "knife", "glass"]

# Default depth assumptions for volume estimation
VOLUME_DEFAULTS = {
    "cup": {"shape": "cylinder", "depth_mm": 100},
    "bottle": {"shape": "cylinder", "depth_mm": 200},
    "glass": {"shape": "cylinder", "depth_mm": 120},
    "mug": {"shape": "cylinder", "depth_mm": 95},
    "bowl": {"shape": "cylinder", "depth_mm": 70},
    "plate": {"shape": "cylinder", "depth_mm": 20},
    "pan": {"shape": "cylinder", "depth_mm": 50},
}

# Paths
PRED_DIR = "runs/detect/predict/labels"
IMG_DIR = "runs/detect/predict"
OUT_DIR = "outputs/yolo_baseline/reports"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "overlays"), exist_ok=True)

def load_class_names(yaml_path="configs/coco_synthetic.yaml"):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return names

def parse_yolo_txt(txt_path):
    dets = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None
            dets.append((cls_id, xc, yc, w, h, conf))
    return dets

def yolo_to_pixel_bbox(xc, yc, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    x = (xc * img_w) - bw / 2
    y = (yc * img_h) - bh / 2
    return int(x), int(y), int(bw), int(bh)

def compute_scale_mm_per_px(ref_name, bw_px, bh_px):
    ref = REFERENCE_SIZES_MM.get(ref_name)
    if not ref:
        return None
    if ref["type"] == "circle":
        d_mm = ref["diameter_mm"]
        d_px = max(bw_px, bh_px)
        return d_mm / d_px if d_px > 0 else None
    elif ref["type"] == "cylinder":
        d_mm = ref["diameter_mm"]
        d_px = max(bw_px, bh_px)
        return d_mm / d_px if d_px > 0 else None
    return None

def estimate_volume(name, w_mm, h_mm):
    if name not in VOLUME_DEFAULTS:
        return None
    depth = VOLUME_DEFAULTS[name]["depth_mm"]
    r = w_mm / 2
    return math.pi * (r**2) * depth

def draw_overlay(img_path, dets_info, out_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = ImageFont.load_default()
    for d in dets_info:
        x, y, w, h = d["bbox_xywh_px"]
        color = (0, 255, 0) if d.get("is_ref") else (255, 165, 0)
        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)
        text = d["label"]
        if d.get("dim_mm"):
            text += f" | {d['dim_mm']}"
        if d.get("volume_ml"):
            text += f" | {d['volume_ml']}"
        draw.text((x + 3, y + 3), text, fill=color, font=font)
    img.save(out_path)

def main():
    class_names = load_class_names()
    label_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.txt")))
    metrics = {"images_processed": 0, "images_with_scale": 0, "per_image": {}}

    for txt_path in label_files:
        base = Path(txt_path).stem
        img_path = os.path.join(IMG_DIR, base + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMG_DIR, base + ".png")
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size
        dets = parse_yolo_txt(txt_path)

        scales, dets_info = [], []
        for (cls_id, xc, yc, w, h, conf) in dets:
            name = class_names[cls_id]
            x_px, y_px, bw_px, bh_px = yolo_to_pixel_bbox(xc, yc, w, h, img_w, img_h)
            info = {"class_name": name, "bbox_xywh_px": (x_px, y_px, bw_px, bh_px), "conf": conf}

            if name in REFERENCE_SIZES_MM:
                scale = compute_scale_mm_per_px(name, bw_px, bh_px)
                if scale:
                    scales.append(scale)
                info["is_ref"] = True
                info["label"] = f"{name} (anchor)"
            else:
                info["is_ref"] = False
                info["label"] = name
            dets_info.append(info)

        metrics["images_processed"] += 1
        scale_mm_per_px = None
        if scales:
            scale_mm_per_px = sorted(scales)[len(scales)//2]
            metrics["images_with_scale"] += 1

        per_image_summary = {"image": img_path, "scale_mm_per_px": scale_mm_per_px, "detections": []}
        for info in dets_info:
            x, y, bw, bh = info["bbox_xywh_px"]
            name = info["class_name"]
            det_entry = {"class": name, "bbox_xywh_px": [x, y, bw, bh], "conf": info.get("conf")}
            if scale_mm_per_px and (name in TARGET_CLASSES or name in VOLUME_DEFAULTS):
                w_mm = bw * scale_mm_per_px
                h_mm = bh * scale_mm_per_px
                info["dim_mm"] = f"{w_mm:.1f}×{h_mm:.1f} mm"
                vol = estimate_volume(name, w_mm, h_mm)
                if vol:
                    info["volume_ml"] = f"{vol/1000:.1f} ml"
                    det_entry["volume_ml"] = vol/1000
            per_image_summary["detections"].append(det_entry)

        metrics["per_image"][base] = per_image_summary
        overlay_path = os.path.join(OUT_DIR, "overlays", base + "_overlay.jpg")
        draw_overlay(img_path, dets_info, overlay_path)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Calibration complete. Processed {metrics['images_processed']} images, "
          f"{metrics['images_with_scale']} with scale anchors.")

if __name__ == "__main__":
    main()
