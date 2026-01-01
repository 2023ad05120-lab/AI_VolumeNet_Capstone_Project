import os
import glob

# List of IDs you want to check
missing_ids = [40, 50, 55, 56, 57, 59, 62, 70, 71, 86, 87,
               107, 108, 114, 129, 139, 143, 150, 151, 154,
               163, 166, 168, 173, 181, 185, 188, 189, 191, 197]

# Path to your ground truth folder
gt_root = r"data\mendeley_depth\val\depth"

def scan_ids(gt_root, ids):
    gt_files = glob.glob(os.path.join(gt_root, "**", "*.png"), recursive=True)
    print(f"Total ground truth files found: {len(gt_files)}")

    for pred_id in ids:
        matches = [f for f in gt_files if str(pred_id) in os.path.basename(f)]
        if matches:
            print(f"\nID {pred_id} → Found {len(matches)} matches:")
            for m in matches[:10]:  # show first 10 matches
                print("   ", m)
        else:
            print(f"\nID {pred_id} → No matches found")

if __name__ == "__main__":
    scan_ids(gt_root, missing_ids)
