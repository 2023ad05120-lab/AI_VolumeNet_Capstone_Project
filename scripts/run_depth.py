# scripts/run_depth.py
import torch
import cv2
import os
import numpy as np

IMG_DIR = "data/processed/coco/images/val"
OUT_DIR = "runs/depth/val"
os.makedirs(OUT_DIR, exist_ok=True)

# Load MiDaS model from Torch Hub
model_type = "DPT_Large"  # options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
midas = torch.hub.load("isl-org/MiDaS", model_type)
midas.eval()

# Load transforms
transform = torch.hub.load("isl-org/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = transform.dpt_transform
else:
    transform = transform.small_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

for fname in os.listdir(IMG_DIR):
    if not fname.endswith(".jpg"):
        continue
    img_path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    out_path = os.path.join(OUT_DIR, fname.replace(".jpg", ".png"))
    cv2.imwrite(out_path, depth_uint8)

print(f"✅ Depth maps saved to {OUT_DIR}")
