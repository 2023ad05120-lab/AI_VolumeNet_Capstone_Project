import os
import glob
import argparse
import cv2
import numpy as np
import json
import re
import csv

def to_single_channel(img):
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 4:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return img

def scale_calibrate(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    if pred.max() > pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)
        pred = pred * (gt.max() - gt.min()) + gt.min()
    return pred, gt

def compute_metrics(pred, gt):
    pred = to_single_channel(pred)
    gt = to_single_channel(gt)
    if gt.shape != pred.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred, gt = scale_calibrate(pred, gt)
    mask = (gt > 0)
    pred = pred[mask]
    gt = gt[mask]
    if pred.size == 0 or gt.size == 0:
        return None
    eps = 1e-6
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    mae = np.mean(np.abs(pred - gt))
    thresh = np.maximum(gt / (pred + eps), pred / (gt + eps))
    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25 ** 2)
    delta3 = np.mean(thresh < 1.25 ** 3)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3),
    }

def save_visual(pred, gt, out_path):
    if gt.shape != pred.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_vis = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gt_vis = cv2.normalize(gt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if len(pred_vis.shape) == 3:
        pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_BGR2GRAY)
    if len(gt_vis.shape) == 3:
        gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(pred_vis, gt_vis)
    vis = np.hstack([pred_vis, gt_vis, diff])
    cv2.imwrite(out_path, vis)

def find_gt_match(pred_id, gt_files):
    pattern = rf"[\\/]{int(pred_id)}\.png$"
    for f in gt_files:
        if re.search(pattern, f):
            return f
    return None

def main(args):
    pred_files = sorted(glob.glob(os.path.join(args.pred, "*.png")))
    gt_files = glob.glob(os.path.join(args.gt, "**", "*.png"), recursive=True)

    print("----- DEBUG SUMMARY -----")
    print(f"Total predictions found: {len(pred_files)}")
    print(f"Total ground truth files found: {len(gt_files)}")
    print(f"Example prediction filenames: {[os.path.basename(f) for f in pred_files[:5]]}")
    print(f"Example ground truth filenames: {[os.path.basename(f) for f in gt_files[:5]]}")
    print("-------------------------")

    metrics_list = []
    samples_evaluated = 0

    # ✅ Always resolve to main outputs/evaluation folder
    vis_dir = os.path.abspath(os.path.join("outputs", "evaluation", "vis"))
    os.makedirs(vis_dir, exist_ok=True)

    csv_path = os.path.abspath(os.path.join("outputs", "evaluation", "per_sample_metrics.csv"))
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # ✅ Proper header row
        csv_writer.writerow([
            "prediction_file", "ground_truth_file",
            "rmse", "mae", "delta1", "delta2", "delta3"
        ])

        for idx, pred_path in enumerate(pred_files):
            fname = os.path.basename(pred_path)
            digits = ''.join([c for c in fname if c.isdigit()])
            pred_id = str(int(digits))

            gt_path = find_gt_match(pred_id, gt_files)
            if gt_path is None:
                print(f"⚠️ No ground truth found for {fname} (ID {pred_id})")
                continue

            if idx < 10:
                print(f"Pair {idx+1}: {fname} <-> {os.path.basename(gt_path)}")

            pred_img = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

            if pred_img is None or gt_img is None:
                print(f"⚠️ Skipped {fname} (could not load)")
                continue

            metrics = compute_metrics(pred_img, gt_img)
            if metrics:
                metrics_list.append(metrics)
                samples_evaluated += 1
                # ✅ Preserve gesture folder info
                csv_writer.writerow([
                    fname,
                    os.path.relpath(gt_path, args.gt),
                    metrics["rmse"], metrics["mae"],
                    metrics["delta1"], metrics["delta2"], metrics["delta3"]
                ])

                if samples_evaluated <= 5:
                    out_path = os.path.join(vis_dir, f"vis_{fname}")
                    save_visual(to_single_channel(pred_img), to_single_channel(gt_img), out_path)

    # ✅ Aggregate metrics
    if samples_evaluated > 0:
        rmse_mean = float(np.mean([m["rmse"] for m in metrics_list]))
        mae_mean = float(np.mean([m["mae"] for m in metrics_list]))
        delta1 = float(np.mean([m["delta1"] for m in metrics_list]))
        delta2 = float(np.mean([m["delta2"] for m in metrics_list]))
        delta3 = float(np.mean([m["delta3"] for m in metrics_list]))
    else:
        rmse_mean = mae_mean = delta1 = delta2 = delta3 = None

    results = {
        "samples_evaluated": samples_evaluated,
        "rmse_mean": rmse_mean,
        "mae_mean": mae_mean,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "per_sample_csv": csv_path
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Folder with predicted depth maps (.png)")
    parser.add_argument("--gt", required=True, help="Folder with ground truth depth maps (.png)")
    parser.add_argument("--output", required=False, help="Output JSON file for metrics")
    args = parser.parse_args()
    main(args)
