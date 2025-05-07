"""
Script to prepare the dataset for training.
"""

import os
import sys
from pathlib import Path
from glob import glob

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from yolo_detector.data_utils import (
    count_classes_in_label_file,
    move_files,
    reload_and_save_images,
    stratified_split
)

def main(input_dir: str, output_dir: str):
    """
    Prepare the dataset for training.
    
    Args:
        input_dir: Directory containing raw images and labels
        output_dir: Directory to save the prepared dataset
    """
    # Configuration
    raw_images_dir = os.path.join(input_dir, "raw_images")
    raw_labels_dir = os.path.join(input_dir, "raw_labels")
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    # Create output directories
    for split in ("train", "val"):
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # Step 1: Split dataset into train/val sets
    print("\nStep 1: Splitting dataset...")
    train_files, val_files, total_class_counts = stratified_split(raw_images_dir, raw_labels_dir)
    move_files(train_files, raw_images_dir, raw_labels_dir, 
              os.path.join(images_dir, "train"), os.path.join(labels_dir, "train"))
    move_files(val_files, raw_images_dir, raw_labels_dir,
              os.path.join(images_dir, "val"), os.path.join(labels_dir, "val"))
    
    print(f"Total images: {len(train_files) + len(val_files)}")
    print(f"Train images: {len(train_files)}")
    print(f"Val images:   {len(val_files)}")
    
    # Step 2: Fix any corrupted images
    print("\nStep 2: Fixing corrupted images...")
    n_fixed_train = reload_and_save_images(os.path.join(images_dir, "train"))
    n_fixed_val = reload_and_save_images(os.path.join(images_dir, "val"))
    print(f"Fixed {n_fixed_train} training images and {n_fixed_val} validation images")
    
    print("\nDataset preparation complete!")
    print(f"Prepared dataset saved to: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_dataset.py <input_dir> <output_dir>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2]) 