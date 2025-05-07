"""
Utility functions for handling dataset operations.
"""

import os
import random
import shutil
from PIL import Image, ImageFile
from collections import Counter

ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow truncated image loading

def count_classes_in_label_file(label_path: str) -> Counter:
    """
    Count how many instances of each class index appear in a YOLO label file.

    Parameters
    ----------
    label_path : str
        Path to a `.txt` file whose lines follow the YOLO format:
        ``<class_id> x_center y_center width height``

    Returns
    -------
    collections.Counter
        Mapping ``class_id -> instance_count``.
    """
    class_counts = Counter()
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:          # valid YOLO row
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
    except Exception as e:                  # file missing / unreadable
        print(f"Error reading {label_path}: {e}")
    return class_counts

def stratified_split(raw_images_dir, raw_labels_dir, split_ratio=0.8):
    image_files = [
        f for f in os.listdir(raw_images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    total_class_counts: Counter[int] = Counter()
    for img_file in image_files:
        name, _ = os.path.splitext(img_file)
        label_path = os.path.join(raw_labels_dir, name + ".txt")
        if os.path.exists(label_path):
            total_class_counts.update(count_classes_in_label_file(label_path))

    class_to_images: dict[int, list[str]] = {cid: [] for cid in total_class_counts}
    for img_file in image_files:
        name, _ = os.path.splitext(img_file)
        lbl_path = os.path.join(raw_labels_dir, name + ".txt")
        if os.path.exists(lbl_path):
            for cid in count_classes_in_label_file(lbl_path):
                class_to_images[cid].append(img_file)

    import random
    random.seed(42)
    image_assignments = {}
    all_images = list(image_files)
    random.shuffle(all_images)
    for img_file in all_images:
        name, _ = os.path.splitext(img_file)
        lbl_path = os.path.join(raw_labels_dir, name + ".txt")
        if not os.path.exists(lbl_path):
            continue
        classes_in_image = list(count_classes_in_label_file(lbl_path).keys())
        train_vote = 0
        val_vote = 0
        for cls in classes_in_image:
            cls_train = sum([1 for img in image_assignments if cls in count_classes_in_label_file(os.path.join(raw_labels_dir, os.path.splitext(img)[0] + ".txt")) and image_assignments[img] == 'train'])
            cls_val = sum([1 for img in image_assignments if cls in count_classes_in_label_file(os.path.join(raw_labels_dir, os.path.splitext(img)[0] + ".txt")) and image_assignments[img] == 'val'])
            cls_total = cls_train + cls_val
            if cls_total == 0:
                train_vote += 1
            else:
                train_ratio = cls_train / cls_total
                if train_ratio < split_ratio:
                    train_vote += 1
                else:
                    val_vote += 1
        if train_vote >= val_vote:
            image_assignments[img_file] = 'train'
        else:
            image_assignments[img_file] = 'val'
    train_files = [img for img, split in image_assignments.items() if split == 'train']
    val_files = [img for img, split in image_assignments.items() if split == 'val']
    return train_files, val_files, total_class_counts

def move_files(file_list: list[str], img_src: str, lbl_src: str, img_dst: str, lbl_dst: str) -> None:
    """
    Copy images and their YOLO label files to destination folders.

    Parameters
    ----------
    file_list : list[str]
        Filenames (with extension) to move.
    img_src : str
        Source directory for images.
    lbl_src : str
        Source directory for labels.
    img_dst : str
        Directory to receive images.
    lbl_dst : str
        Directory to receive label `.txt` files.
    """
    for file in file_list:
        name, _ = os.path.splitext(file)
        img_path = os.path.join(img_src, file)
        lbl_path = os.path.join(lbl_src, name + ".txt")

        if os.path.exists(img_path) and os.path.exists(lbl_path):
            shutil.copy(img_path, os.path.join(img_dst, file))
            shutil.copy(lbl_path, os.path.join(lbl_dst, name + ".txt"))

def reload_and_save_images(folder_path: str) -> int:
    """
    Re‑encode every image inside *folder_path* to RGB and overwrite it in place.

    Parameters
    ----------
    folder_path : str
        Directory that holds the images to repair.

    Returns
    -------
    int
        Count of images successfully rewritten.
    """
    fixed_count = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, filename)

            try:
                img = Image.open(path)
                img = img.convert("RGB")
                img.save(path, optimize=True)
                fixed_count += 1
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    return fixed_count