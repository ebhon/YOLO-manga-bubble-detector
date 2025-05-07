"""
Functions for running inference with the trained YOLO model.
"""

import os
import shutil
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from .postprocessing import apply_post_processing_rules
from .data_utils import reload_and_save_images

def prepare_test_dir(base_dir, raw_images_dir, val_files, max_samples=5):
    """
    Prepare test directory with sample images.
    
    Args:
        base_dir: Base directory path
        raw_images_dir: Directory containing raw images
        val_files: List of validation files to use
        max_samples: Maximum number of samples to copy
        
    Returns:
        Path to the test directory
    """
    test_dir = os.path.join(base_dir, "test_set")
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory {test_dir} does not exist. Creating it.")
        os.makedirs(test_dir, exist_ok=True)
        for i, file in enumerate(val_files[:max_samples]):
            shutil.copy(os.path.join(raw_images_dir, file), test_dir)
    return test_dir

def draw_detections(image_path: str, detections: list, output_path: str) -> None:
    # Class mapping
    class_names = {
        0: "bubble",
        1: "narration",
        2: "other",
        3: "text",
        4: "ui"
    }
    class_colors = {
        "bubble": (255, 0, 0),      # Blue
        "narration": (0, 255, 255), # Yellow
        "other": (0, 0, 255),       # Red
        "text": (0, 255, 0),        # Green
        "ui": (255, 0, 255),        # Magenta
    }
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    for det in detections:
        x, y = int(det['x']), int(det['y'])
        w, h = int(det['width']), int(det['height'])
        conf = det['confidence']
        class_idx = det.get('class', 0)
        class_name = class_names.get(class_idx, str(class_idx))
        color = class_colors.get(class_name, (0, 255, 0))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imwrite(output_path, img)

def run_folder_inference(model_path, test_dir, save_dir="predictions/test_set"):
    model = YOLO(model_path)
    reload_and_save_images(test_dir)
    results = model.predict(test_dir, save=False)
    all_processed = apply_post_processing_rules(results)
    os.makedirs(save_dir, exist_ok=True)
    for result, dets in zip(results, all_processed):
        img_path = result.path
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(save_dir, f"processed_{base_name}.jpg")
        draw_detections(img_path, dets, output_path)
        print(f"Saved processed image to: {output_path}")
    print("Inference complete with post-processing rules applied")

def print_detections(detections: list) -> None:
    """
    Print detection results in a formatted way.
    
    Args:
        detections: List of detections from run_inference
    """
    if not detections:
        print("No detections found")
        return
    
    print("\nDetection Results:")
    for i, det in enumerate(detections, 1):
        print(f"\nDetection {i}:")
        print(f"  Class: {det['class']}")
        print(f"  Confidence: {det['confidence']:.2f}")
        print(f"  Position: ({det['x']:.1f}, {det['y']:.1f})")
        print(f"  Size: {det['width']:.1f}x{det['height']:.1f}")