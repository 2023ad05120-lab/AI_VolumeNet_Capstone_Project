import os

# Paths to prediction and ground truth folders
pred_dir = r"C:\Users\khunz\AI_VolumeNet_Capstone_Project\AI_VolumeNet_Capstone_Project\runs\depth\val"
gt_dir = r"C:\Users\khunz\AI_VolumeNet_Capstone_Project\AI_VolumeNet_Capstone_Project\data\mendeley_depth\val"

# Get filenames (without extension)
pred_files = {os.path.splitext(f)[0] for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg'))}
gt_files = {os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg'))}

# Compare sets
missing_in_gt = pred_files - gt_files
missing_in_pred = gt_files - pred_files
matched = pred_files & gt_files

# Report
print(f"✅ Matched samples: {len(matched)}")
print(f"❌ Missing in ground truth: {len(missing_in_gt)}")
if missing_in_gt:
    print("  →", sorted(missing_in_gt)[:10])

print(f"❌ Missing in predictions: {len(missing_in_pred)}")
if missing_in_pred:
    print("  →", sorted(missing_in_pred)[:10])
