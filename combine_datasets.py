import os
import shutil
import pandas as pd
import random
from pathlib import Path

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "data"
COMBINED_DIR = os.path.join(DATA_DIR, "combined_dataset")

# Source datasets
DATASETS = {
    
    "dogs_skin_diseases": {
        "type": "folder",  # images are in train/test/valid subfolders
        "image_root": DATA_DIR + "/Dog's skin diseases"
    },
    "pet_disease": {
        "type": "folder",
        "image_root": DATA_DIR + "/pet_disease/data"
    }
}

# Train/val/test split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def normalize_label(label):
    """Normalize disease labels to lowercase, remove spaces/underscores"""
    return label.strip().lower().replace(" ", "_").replace("-", "_")

def copy_image(src_path, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src_path, dst_dir)

# -----------------------------
# STEP 1: Gather all images and labels
# -----------------------------
all_images = []  # list of tuples (image_path, label)



# Dataset 2 and 3: Folder-based
for key in ["dogs_skin_diseases", "pet_disease"]:
    dataset = DATASETS[key]
    if dataset["type"] == "folder":
        for root, dirs, files in os.walk(dataset["image_root"]):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    # Use parent folder as label (common convention)
                    label = normalize_label(Path(root).name)
                    img_path = os.path.join(root, file)
                    all_images.append((img_path, label))

print(f"Total images collected: {len(all_images)}")

# -----------------------------
# STEP 2: Shuffle and split
# -----------------------------
random.shuffle(all_images)
n_total = len(all_images)
n_train = int(TRAIN_RATIO * n_total)
n_val = int(VAL_RATIO * n_total)
n_test = n_total - n_train - n_val

train_imgs = all_images[:n_train]
val_imgs = all_images[n_train:n_train+n_val]
test_imgs = all_images[n_train+n_val:]

splits = {
    "train": train_imgs,
    "val": val_imgs,
    "test": test_imgs
}

# -----------------------------
# STEP 3: Copy images to combined_dataset and create CSV
# -----------------------------
annotations = []

for split_name, images in splits.items():
    for img_path, label in images:
        dst_dir = os.path.join(COMBINED_DIR, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(img_path))
        shutil.copy(img_path, dst_path)
        annotations.append([os.path.relpath(dst_path, COMBINED_DIR), label, split_name])

# Save CSV
annotations_df = pd.DataFrame(annotations, columns=["filename", "label", "split"])
annotations_df.to_csv(os.path.join(COMBINED_DIR, "annotations.csv"), index=False)

print(f"Combined dataset created at: {COMBINED_DIR}")
print(f"CSV saved at: {os.path.join(COMBINED_DIR, 'annotations.csv')}")
print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
