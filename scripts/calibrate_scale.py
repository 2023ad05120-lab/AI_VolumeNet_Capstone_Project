import os
import glob
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Update these to match your class names
REFERENCE_SIZES_MM = {
    "coin": {"type": "circle", "diameter_mm": 24.0},         # edit to your coin
    "card": {"type": "rect", "width_mm": 85.6, "height_mm": 53.98},
    "a4_sheet": {"type": "rect", "width_mm": 210.0, "height_mm": 297.0},
}

# Classes to calibrate for (non-reference)
TARGET_CLASSES = ["box", "parcel", "object"]  # edit to your dataset

# Location of YOLO predictions (*.txt) and images
PRED_DIR = "runs/detect/predict/labels"       # or wherever save_txt outputs
IMG_DIR = "runs/detect/predict"               # matching folder of images

OUT_DIR = "outputs/yolo_baseline/reports"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "overlays"), exist_ok=True)

def load_class_names(yaml_path="configs/coco_synthetic.yaml"):
    # Fallback: you can hardcode names if needed
    import yaml
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # names may be list or dict; normalize to list index by class_id
    names = data.get("names", [])
    if isinstance(names, dict):
        # convert to list by key order
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return names

def parse_yolo_txt(txt_path):
    # Returns list of detections: [class_id, x_c, y_c, w, h, conf (optional)]
    dets = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # YOLO save_txt may include conf as 6th value; handle both cases
            class_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None
            dets.append((class_id, x_c, y_c, w, h, conf))
    return dets

def yolo_to_pixel_bbox(xc, yc, w, h, img_w, img_h):
    # Convert normalized YOLO bbox to pixel coords and sizes
    bw = w * img_w
    bh = h * img_h
    x = (xc * img_w) - bw / 2
    y = (yc * img_h) - bh / 2
    return int(x), int(y), int(bw), int(bh)

def compute_scale_mm_per_px(ref_name, bbox_w_px, bbox_h_px):
    ref = REFERENCE_SIZES_MM.get(ref_name)
    if not ref:
        return None
    if ref["type"] == "circle":
        # coin: use width as diameter proxy (choose max of w/h to be safe)
        diameter_mm = ref["diameter_mm"]
        d_px = max(bbox_w_px, bbox_h_px)
        return diameter_mm / d_px if d_px > 0 else None
    elif ref["type"] == "rect":
        # card/A4: use longer side for stability
        longer_mm = max(ref["width_mm"], ref["height_mm"])
        longer_px = max(bbox_w_px, bbox_h_px)
        return longer_mm / longer_px if longer_px > 0 else None
    return None

def draw_overlay(img_path, dets_info, class_names, out_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = ImageFont.load_default()
    for d in dets_info:
        x, y, w, h = d["bbox_xywh_px"]
        label = d["label"]
        color = (0, 255, 0) if d.get("is_ref") else (255, 165, 0)
        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)
        text = label
        if d.get("dim_mm"):
            text += f" | {d['dim_mm']}"
        draw.text((x + 3, y + 3), text, fill=color, font=font)
    img.save(out_path)

def main():
    class_names = load_class_names()
    inv_class = {name: idx for idx, name in enumerate(class_names)}

    label_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.txt")))
    metrics = {
        "images_processed": 0,
        "images_with_scale": 0,
        "scales_mm_per_px": [],
        "per_image": {}
    }

    for txt_path in label_files:
        base = Path(txt_path).stem
        # predictions save image as *.jpg by default; adapt if png
        img_path_candidates = [
            os.path.join(IMG_DIR, base + ".jpg"),
            os.path.join(IMG_DIR, base + ".png"),
        ]
        img_path = next((p for p in img_path_candidates if os.path.exists(p)), None)
        if not img_path:
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size
        dets = parse_yolo_txt(txt_path)

        # Collect reference scales from ref detections
        scales = []
        dets_info = []
        for (cls_id, xc, yc, w, h, conf) in dets:
            name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            x_px, y_px, bw_px, bh_px = yolo_to_pixel_bbox(xc, yc, w, h, img_w, img_h)

            info = {
                "class_id": cls_id,
                "class_name": name,
                "bbox_xywh_px": (x_px, y_px, bw_px, bh_px),
                "conf": conf,
            }

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
        if scales:
            # Robust scale per image
            scales_sorted = sorted(scales)
            scale_mm_per_px = scales_sorted[len(scales_sorted) // 2]
            metrics["images_with_scale"] += 1
            metrics["scales_mm_per_px"].append(scale_mm_per_px)
        else:
            scale_mm_per_px = None

        # Apply scale to targets and annotate
        per_image_summary = {
            "image": img_path,
            "scale_mm_per_px": scale_mm_per_px,
            "detections": []
        }

        for info in dets_info:
            x, y, bw, bh = info["bbox_xywh_px"]
            name = info["class_name"]
            det_entry = {
                "class": name,
                "bbox_xywh_px": [x, y, bw, bh],
                "conf": info.get("conf"),
            }
            if scale_mm_per_px and (name in TARGET_CLASSES or name not in REFERENCE_SIZES_MM):
                w_mm = bw * scale_mm_per_px
                h_mm = bh * scale_mm_per_px
                info["dim_mm"] = f"{w_mm:.1f}×{h_mm:.1f} mm"
                det_entry["dimensions_mm"] = {"w_mm": w_mm, "h_mm": h_mm}
            per_image_summary["detections"].append(det_entry)

        metrics["per_image"][base] = per_image_summary

        # Draw overlay
        overlay_path = os.path.join(OUT_DIR, "overlays", base + "_overlay.jpg")
        draw_overlay(img_path, dets_info, class_names, overlay_path)

    # Save metrics JSON
    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Calibration complete.\n- Images processed: {metrics['images_processed']}\n- Images with scale: {metrics['images_with_scale']}\n- Saved: {metrics_path}\n- Overlays: {os.path.join(OUT_DIR, 'overlays')}")

if __name__ == "__main__":
    main()
