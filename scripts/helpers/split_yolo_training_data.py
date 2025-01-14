import os
import random
import shutil

# Define paths
images_path = "yolo/data/images"
labels_path = "yolo/data/labels"
output_path = "yolo/data"

# Create directories for train, val, and test splits
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(images_path, split), exist_ok=True)
    os.makedirs(os.path.join(labels_path, split), exist_ok=True)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Get list of all images and corresponding labels
images = [f for f in os.listdir(images_path) if f.endswith((".png", ".jpg", ".jpeg"))]
random.shuffle(images)

# Split data
train_count = int(len(images) * train_ratio)
val_count = int(len(images) * val_ratio)

train_images = images[:train_count]
val_images = images[train_count : train_count + val_count]
test_images = images[train_count + val_count :]


# Helper function to move files
def move_files(file_list, split):
    for file in file_list:
        image_src = os.path.join(images_path, file)
        label_src = os.path.join(
            labels_path,
            file.replace(".jpg", ".txt")
            .replace(".png", ".txt")
            .replace(".jpeg", ".txt"),
        )

        image_dst = os.path.join(images_path, split, file)
        label_dst = os.path.join(labels_path, split, os.path.basename(label_src))

        if os.path.exists(label_src):
            shutil.move(image_src, image_dst)
            shutil.move(label_src, label_dst)
        else:
            print(f"Label for {file} not found, skipping.")


# Move files to corresponding directories
move_files(train_images, "train")
move_files(val_images, "val")
move_files(test_images, "test")


# Function to create .txt files with image paths
def create_txt_file(file_list, split):
    with open(os.path.join(output_path, f"{split}.txt"), "w") as f:
        for file in file_list:
            f.write(f"{os.path.join(images_path, split, file)}\n")


# Create train.txt, val.txt, and test.txt
create_txt_file(train_images, "train")
create_txt_file(val_images, "val")
create_txt_file(test_images, "test")

print("Data split and .txt files created successfully.")
