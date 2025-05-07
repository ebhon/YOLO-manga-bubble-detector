"""
Script to repair corrupted images in the dataset.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from yolo_detector.data_utils import reload_and_save_images

def main():
    # Configuration
    base_dir = "data"
    images_dir = os.path.join(base_dir, "images")
    
    # Repair training images
    print("Repairing training images...")
    n_fixed_train = reload_and_save_images(os.path.join(images_dir, "train"))
    print(f"Fixed {n_fixed_train} training images")
    
    # Repair validation images
    print("\nRepairing validation images...")
    n_fixed_val = reload_and_save_images(os.path.join(images_dir, "val"))
    print(f"Fixed {n_fixed_val} validation images")

if __name__ == "__main__":
    main()