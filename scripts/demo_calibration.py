import torch
import cv2
import numpy as np

# Load MiDaS model from torch.hub
model_type = "DPT_Large"  # options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type.startswith("DPT") else midas_transforms.small_transform

# Load image
img = cv2.imread("data/sample/mug.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img_rgb).unsqueeze(0)

# Predict depth
with torch.no_grad():
    prediction = midas(input_batch)
    depth_map = prediction.squeeze().cpu().numpy()

# Normalize for visualization
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
cv2.imshow("Depth Map", depth_map)
cv2.waitKey(0)

# Example calibration: assume known mug height = 100 mm
pixels_height = depth_map.shape[0]
pixels_per_mm = pixels_height / 100.0
print(f"Calibration factor: {pixels_per_mm:.2f} pixels per mm")
