"""
Script to split the dataset into training and validation sets.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from yolo_detector.data_utils import stratified_split, move_files

def main():
    # Configuration
    base_dir = "data"
    raw_images_dir = os.path.join(base_dir, "raw_images")
    raw_labels_dir = os.path.join(base_dir, "raw_labels")
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    
    # Create train/val directories
    for split in ("train", "val"):
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # Split dataset
    print("Splitting dataset...")
    train_files, val_files, total_class_counts = stratified_split(raw_images_dir, raw_labels_dir)
    
    # Move files to their respective directories
    move_files(train_files, raw_images_dir, raw_labels_dir, 
              os.path.join(images_dir, "train"), os.path.join(labels_dir, "train"))
    move_files(val_files, raw_images_dir, raw_labels_dir,
              os.path.join(images_dir, "val"), os.path.join(labels_dir, "val"))
    
    # Print statistics
    print(f"Total images: {len(train_files) + len(val_files)}")
    print(f"Train images: {len(train_files)}")
    print(f"Val images:   {len(val_files)}")
    print("\nClass distribution:")
    for class_id, count in total_class_counts.items():
        print(f"Class {class_id}: {count}")

if __name__ == "__main__":
    main()