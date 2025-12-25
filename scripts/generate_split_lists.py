import os

def generate_split_lists(images_root="data/processed/coco/images",
                         train_list="data/processed/coco/train.txt",
                         val_list="data/processed/coco/val.txt"):
    train_img_dir = os.path.join(images_root, "train")
    val_img_dir = os.path.join(images_root, "val")

    # Collect image paths
    train_files = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir)
                   if f.lower().endswith((".jpg", ".png"))]
    val_files = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir)
                 if f.lower().endswith((".jpg", ".png"))]

    # Write train.txt
    with open(train_list, "w") as f:
        for path in sorted(train_files):
            f.write(path + "\n")

    # Write val.txt
    with open(val_list, "w") as f:
        for path in sorted(val_files):
            f.write(path + "\n")

    print(f"✅ Generated {train_list} ({len(train_files)} images)")
    print(f"✅ Generated {val_list} ({len(val_files)} images)")

if __name__ == "__main__":
    generate_split_lists()
