import os
import cv2

# Path to your ground truth folder
gt_dir = r"C:\Users\khunz\AI_VolumeNet_Capstone_Project\AI_VolumeNet_Capstone_Project\data\mendeley_depth\val"

for fname in os.listdir(gt_dir):
    if fname.lower().endswith(".png"):
        path = os.path.join(gt_dir, fname)
        img = cv2.imread(path)  # loads the JPEG data (even though extension is .png)
        if img is None:
            print(f"⚠️ Could not load {fname}")
            continue
        # Re‑encode properly as PNG
        cv2.imwrite(path, img)
        print(f"Re‑encoded {fname} as true PNG")

print("✅ Conversion complete. All files are now proper PNGs.")
