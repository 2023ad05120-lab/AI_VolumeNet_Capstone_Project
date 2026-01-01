import os

pred_dir = r"runs/depth/val"
gt_dir   = r"data/mendeley_depth/val/depth_flat"

pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".png")])

for pred_fname, gt_fname in zip(pred_files, sorted(os.listdir(gt_dir))):
    gt_path = os.path.join(gt_dir, gt_fname)
    new_path = os.path.join(gt_dir, pred_fname)
    os.rename(gt_path, new_path)
    print(f"Renamed {gt_fname} → {pred_fname}")

print("✅ Ground truth filenames now aligned with predictions.")
