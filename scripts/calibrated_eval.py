import os, json, csv, math, argparse
from PIL import Image
import numpy as np 

def bbox_from_mask_np(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def normalize_image_id(img_name):
    base = os.path.splitext(os.path.basename(img_name))[0]
    return base.replace("img_", "")

def main(data_dir, anchors_path, output_path):
    # Load metadata
    meta_csv = os.path.join(data_dir, "metadata.csv")
    with open(meta_csv) as f:
        rows = list(csv.DictReader(f))

    # Load anchor dimensions (JSON: { "plate": 26.0, "pan": 28.0, "mug": 8.0 })
    with open(anchors_path) as f:
        anchors = json.load(f)

    errors_w, errors_h, vol_rel_errors = [], [], []

    for r in rows:
        img_id = normalize_image_id(r["image_id"])
        mask_path = os.path.join(data_dir, "masks", f"mask_{img_id}.png")
        if not os.path.exists(mask_path):
            continue

        m = Image.open(mask_path).convert("L")
        m_np = np.array(m)

        ref_mask = (m_np == 0).astype(np.uint8)
        obj_mask = (m_np == 255).astype(np.uint8)

        ref_bbox = bbox_from_mask_np(ref_mask)
        obj_bbox = bbox_from_mask_np(obj_mask)
        if ref_bbox is None or obj_bbox is None:
            continue

        # Anchor calibration: pick anchor width from JSON
        ref_pix_w = max(1, ref_bbox[2] - ref_bbox[0])
        anchor_cm = anchors.get("plate", 8.56)  # default fallback
        est_ppcm = ref_pix_w / anchor_cm

        proj_w_cm = (obj_bbox[2] - obj_bbox[0]) / est_ppcm
        proj_h_cm = (obj_bbox[3] - obj_bbox[1]) / est_ppcm

        gt_dims = json.loads(r["real_dims"])
        shape = r["shape"]

        if shape in ["box", "packet"]:
            gt_proj_w, gt_proj_h = gt_dims["L"], gt_dims["H"]
            gt_vol = gt_dims["L"] * gt_dims["W"] * gt_dims["H"]
        elif shape in ["cylinder", "can", "bottle"]:
            gt_proj_w, gt_proj_h = gt_dims["D"], gt_dims["H"]
            r_cm = gt_dims["D"] / 2.0
            gt_vol = math.pi * r_cm * r_cm * gt_dims["H"]
        elif shape == "sphere":
            gt_proj_w = gt_proj_h = gt_dims["D"]
            r_cm = gt_dims["D"] / 2.0
            gt_vol = 4.0 / 3.0 * math.pi * r_cm**3
        else:
            continue

        errors_w.append(abs(proj_w_cm - gt_proj_w))
        errors_h.append(abs(proj_h_cm - gt_proj_h))

        if shape in ["box", "packet"]:
            est_vol = proj_w_cm * proj_h_cm * gt_dims["W"]
        elif shape in ["cylinder", "can", "bottle"]:
            est_vol = math.pi * (proj_w_cm / 2.0) ** 2 * proj_h_cm
        elif shape == "sphere":
            est_vol = 4.0 / 3.0 * math.pi * (proj_w_cm / 2.0) ** 3
        else:
            est_vol = None

        if est_vol and gt_vol > 0:
            vol_rel_errors.append(abs(est_vol - gt_vol) / gt_vol * 100.0)

    results = {
        "samples_evaluated": len(errors_w),
        "proj_width_mae_cm": float(np.mean(errors_w)) if errors_w else None,
        "proj_height_mae_cm": float(np.mean(errors_h)) if errors_h else None,
        "vol_rel_error_mean_pct": float(np.mean(vol_rel_errors)) if vol_rel_errors else None,
        "vol_rel_error_median_pct": float(np.median(vol_rel_errors)) if vol_rel_errors else None,
    }

    print(json.dumps(results, indent=2))
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed/coco")
    parser.add_argument("--anchors", type=str, default="data/anchors.json")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args.data_dir, args.anchors, args.output)
