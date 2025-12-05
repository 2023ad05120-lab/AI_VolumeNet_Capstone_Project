import os
import json
import cv2
import matplotlib.pyplot as plt

DATASET_DIR = "data/synthetic"
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.json")

def check_structure():
    assert os.path.exists(DATASET_DIR), "Dataset directory missing"
    assert os.path.exists(ANNOTATIONS_FILE), "Annotations file missing"
    print("✅ Structure check passed")

def validate_annotations():
    with open(ANNOTATIONS_FILE, "r") as f:
        ann = json.load(f)
    ids = set()
    for img in ann["images"]:
        assert "id" in img and "file_name" in img
        ids.add(img["id"])
    for a in ann["annotations"]:
        assert a["image_id"] in ids, "Annotation references missing image"
        x, y, w, h = a["bbox"]
        assert w > 0 and h > 0, "Invalid bbox dimensions"
    print("✅ Annotation validation passed")

def visualize_samples(n=5):
    with open(ANNOTATIONS_FILE, "r") as f:
        ann = json.load(f)
    for img in ann["images"][:n]:
        path = os.path.join(DATASET_DIR, "images", img["file_name"])
        image = cv2.imread(path)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Sample: {img['file_name']}")
        plt.show()

if __name__ == "__main__":
    check_structure()
    validate_annotations()
    visualize_samples()

