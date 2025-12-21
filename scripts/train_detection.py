# scripts/train_detection.py
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to YOLO model config or pretrained weights")
    parser.add_argument("--data", type=str, required=True, help="Path to training dataset YAML")
    parser.add_argument("--val", type=str, required=True, help="Path to validation dataset YAML")
    parser.add_argument("--out", type=str, required=True, help="Output directory for logs and checkpoints")
    args = parser.parse_args()

    # Load YOLO model (cfg can be 'yolov8n.yaml' or a pretrained weights file)
    model = YOLO(args.cfg)

    # Train
    model.train(
        data=args.data,
        val=args.val,
        project=args.out,
        name="det_baseline",
        epochs=50,
        imgsz=640
    )

if __name__ == "__main__":
    main()
