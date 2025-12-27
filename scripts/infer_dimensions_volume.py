import os, json, cv2
import numpy as np
from pathlib import Path

ALIGNED_DIR = "outputs/aligned/val"
OUT_DIR = "outputs/yolo_baseline/reports"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "overlays"), exist_ok=True)

# Anchors: known real-world dimensions (mm)
ANCHORS = {
    "plate": {"diameter_mm": 240},
    "pan": {"diameter_mm": 280},
    "mug": {"height_mm": 100}
}

def calibrate_scale(detections):
    """Compute mm/px scale using anchors present in detections."""
    scales = []
    for det in detections:
        cls = det["class"]
        x, y, w, h = det["bbox_xywh_px"]
        if cls in ANCHORS:
            if "diameter_mm" in ANCHORS[cls]:
                scales.append(ANCHORS[cls]["diameter_mm"] / max(w, h))
            elif "height_mm" in ANCHORS[cls]:
                scales.append(ANCHORS[cls]["height_mm"] / h)
    return scales

def estimate_volume(cls, w_mm, h_mm, depth_mm):
    """Simple geometric approximations per class."""
    if cls == "mug":
        radius = w_mm / 2
        return np.pi * (radius**2) * h_mm
    elif cls == "plate":
        radius = w_mm / 2
        return np.pi * (radius**2) * 10  # shallow depth assumption
    elif cls == "pan":
        radius = w_mm / 2
        return np.pi * (radius**2) * 50  # shallow cylinder
    else:
        return w_mm * h_mm * depth_mm  # box approx

def main():
    # Pass 1: collect scales from anchor-rich images
    all_scales = []
    for json_file in Path(ALIGNED_DIR).glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
        detections = data["detections"]
        scales = calibrate_scale(detections)
        all_scales.extend(scales)
    global_scale = np.median(all_scales) if all_scales else 1.0
    print(f"📏 Global scale set to {global_scale:.3f} mm/px")

    # Pass 2: apply scale to all images
    metrics = []
    for json_file in Path(ALIGNED_DIR).glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
        detections = data["detections"]

        # Use per-image scale if available, else fallback to global
        scales = calibrate_scale(detections)
        scale = np.median(scales) if scales else global_scale

        for det in detections:
            cls = det["class"]
            x, y, w, h = det["bbox_xywh_px"]
            w_mm, h_mm = w * scale, h * scale
            depth_median = det["depth_stats"]["depth_median"] if det["depth_stats"] else 1
            depth_mm = depth_median * 100  # rough scaling
            volume = estimate_volume(cls, w_mm, h_mm, depth_mm)

            metrics.append({
                "image": data["image"],
                "class": cls,
                "w_mm": round(w_mm, 2),
                "h_mm": round(h_mm, 2),
                "depth_mm": round(depth_mm, 2),
                "volume_ml": round(volume, 1)
            })

            # Overlay
            img_path = os.path.join("runs/detect/predict", data["image"])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                cv2.putText(img, f"{cls}: {round(volume,1)} ml",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.imwrite(os.path.join(OUT_DIR, "overlays", data["image"]), img)

    # Save metrics
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Volume inference complete. Metrics saved to {OUT_DIR}/metrics.json")

if __name__ == "__main__":
    main()
