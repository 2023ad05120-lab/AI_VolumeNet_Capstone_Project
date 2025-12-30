# baseline_eval.py
# Baseline evaluation for the synthetic dataset produced above.
# Usage: python baseline_eval.py --data_dir data/processed/coco --output outputs/evaluation/detection_metrics.json

import os
import json
import csv
import math
import argparse
from PIL import Image
import numpy as np

def bbox_from_mask_np(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    return [x0, y0, x1, y1]

def normalize_image_id(img_name):
    # Example: img_0002.png -> 0002
    base = os.path.splitext(os.path.basename(img_name))[0]
    return base.replace("img_", "")

def main(data_dir, output_path):
    meta_csv = os.path.join(data_dir, "metadata.csv")
    with open(meta_csv) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]

    errors_w, errors_h, vol_rel_errors = [], [], []

    for r in rows:
        img_id = normalize_image_id(r["image_id"])
        mask_path = os.path.join(data_dir, "masks", f"mask_{img_id}.png")

        if not os.path.exists(mask_path):
            continue

        m = Image.open(mask_path).convert("L")
        m_np = np.array(m)

        # Adjusted for binary masks (0 = reference, 255 = object)
        ref_mask = (m_np == 0).astype(np.uint8)
        obj_mask = (m_np == 255).astype(np.uint8)

        ref_bbox = bbox_from_mask_np(ref_mask)
        obj_bbox = bbox_from_mask_np(obj_mask)
        if ref_bbox is None or obj_bbox is None:
            continue

        ref_pix_w = max(1, ref_bbox[2] - ref_bbox[0])
        ref_w_cm = 8.56  # known reference width in cm
        est_ppcm = ref_pix_w / ref_w_cm

        proj_w_cm = (obj_bbox[2] - obj_bbox[0]) / est_ppcm
        proj_h_cm = (obj_bbox[3] - obj_bbox[1]) / est_ppcm

        gt_dims = json.loads(r["real_dims"])
        shape = r["shape"]

        if shape in ["box", "packet"]:
            gt_proj_w = gt_dims["L"]
            gt_proj_h = gt_dims["H"]
            gt_vol = gt_dims["L"] * gt_dims["W"] * gt_dims["H"]
        elif shape in ["cylinder", "can", "bottle"]:
            gt_proj_w = gt_dims["D"]
            gt_proj_h = gt_dims["H"]
            r_cm = gt_dims["D"] / 2.0
            gt_vol = math.pi * r_cm * r_cm * gt_dims["H"]
        elif shape == "sphere":
            gt_proj_w = gt_dims["D"]
            gt_proj_h = gt_dims["D"]
            r_cm = gt_dims["D"] / 2.0
            gt_vol = 4.0 / 3.0 * math.pi * r_cm**3
        else:
            continue

        errors_w.append(abs(proj_w_cm - gt_proj_w))
        errors_h.append(abs(proj_h_cm - gt_proj_h))

        # volume estimate
        if shape in ["box", "packet"]:
            est_vol = proj_w_cm * proj_h_cm * gt_dims["W"]
        elif shape in ["cylinder", "can", "bottle"]:
            est_vol = math.pi * (proj_w_cm / 2.0) ** 2 * proj_h_cm
        elif shape == "sphere":
            est_vol = 4.0 / 3.0 * math.pi * (proj_w_cm / 2.0) ** 3
        else:
            est_vol = None

        if est_vol is not None and gt_vol > 0:
            vol_rel_errors.append(abs(est_vol - gt_vol) / gt_vol * 100.0)

    results = {
        "samples_evaluated": len(errors_w),
        "proj_width_mae_cm": float(np.mean(errors_w)) if errors_w else None,
        "proj_height_mae_cm": float(np.mean(errors_h)) if errors_h else None,
        "vol_rel_error_mean_pct": float(np.mean(vol_rel_errors)) if vol_rel_errors else None,
        "vol_rel_error_median_pct": float(np.median(vol_rel_errors)) if vol_rel_errors else None,
    }

    print("Samples evaluated:", results["samples_evaluated"])
    print("Projected Width MAE (cm):", results["proj_width_mae_cm"])
    print("Projected Height MAE (cm):", results["proj_height_mae_cm"])
    print("Volume Relative Error (%) mean:", results["vol_rel_error_mean_pct"])
    print("Volume Relative Error (%) median:", results["vol_rel_error_median_pct"])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed/coco")
    parser.add_argument("--output", type=str, default=None)
    args, unknown = parser.parse_known_args()
    main(args.data_dir, args.output)
