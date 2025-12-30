# scripts/run_yolo_inference.py
import argparse
import os
import json
from ultralytics import YOLO

def run_inference(image_dir, output_path, model_path):
    model = YOLO(model_path)
    results = model.predict(source=image_dir, save=False, conf=0.25)

    detections = []
    for r in results:
        image_name = os.path.basename(r.path)
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        detections.append({
            "image": image_name,
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "classes": classes.tolist()
        })

    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help='Path to image folder')
    parser.add_argument('--output', required=True, help='Path to save detections.json')
    parser.add_argument('--model', default='yolo11n.pt', help='Path to YOLO model weights')
    args = parser.parse_args()

    run_inference(args.images, args.output, args.model)
