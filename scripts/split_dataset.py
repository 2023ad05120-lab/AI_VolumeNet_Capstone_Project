import os
import shutil
import random

def split_dataset(
    images_dir="data/processed/coco/images",
    masks_dir="data/processed/coco/masks",
    train_ratio=0.8,
    seed=42
):
    random.seed(seed)

    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]
    random.shuffle(image_files)

    # Split into train/val
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # Create output folders
    train_img_dir = os.path.join(images_dir, "train")
    val_img_dir = os.path.join(images_dir, "val")
    train_mask_dir = os.path.join(masks_dir, "train")
    val_mask_dir = os.path.join(masks_dir, "val")

    for d in [train_img_dir, val_img_dir, train_mask_dir, val_mask_dir]:
        os.makedirs(d, exist_ok=True)

    # Move files
    for f in train_files:
        shutil.copy(os.path.join(images_dir, f), os.path.join(train_img_dir, f))
        mask_file = f.replace(".jpg", ".png")  # adjust if mask naming differs
        if os.path.exists(os.path.join(masks_dir, mask_file)):
            shutil.copy(os.path.join(masks_dir, mask_file), os.path.join(train_mask_dir, mask_file))

    for f in val_files:
        shutil.copy(os.path.join(images_dir, f), os.path.join(val_img_dir, f))
        mask_file = f.replace(".jpg", ".png")
        if os.path.exists(os.path.join(masks_dir, mask_file)):
            shutil.copy(os.path.join(masks_dir, mask_file), os.path.join(val_mask_dir, mask_file))

    print(f"✅ Split complete: {len(train_files)} train, {len(val_files)} val")

if __name__ == "__main__":
    split_dataset()
