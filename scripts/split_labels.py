import os
import shutil

def split_labels(images_root="data/processed/coco/images",
                 labels_root="data/processed/coco/labels"):
    # Define train/val image folders
    train_img_dir = os.path.join(images_root, "train")
    val_img_dir = os.path.join(images_root, "val")

    # Define train/val label folders
    train_lbl_dir = os.path.join(labels_root, "train")
    val_lbl_dir = os.path.join(labels_root, "val")
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Move labels according to image split
    for img_file in os.listdir(train_img_dir):
        base = os.path.splitext(img_file)[0]
        lbl_file = f"{base}.txt"
        src = os.path.join(labels_root, lbl_file)
        dst = os.path.join(train_lbl_dir, lbl_file)
        if os.path.exists(src):
            shutil.copy(src, dst)

    for img_file in os.listdir(val_img_dir):
        base = os.path.splitext(img_file)[0]
        lbl_file = f"{base}.txt"
        src = os.path.join(labels_root, lbl_file)
        dst = os.path.join(val_lbl_dir, lbl_file)
        if os.path.exists(src):
            shutil.copy(src, dst)

    print("✅ Labels split into train/ and val/ successfully.")

if __name__ == "__main__":
    split_labels()
