"""
Script to run inference and visualize predictions on test images.

Pipeline:
1. Load the best fine‑tuned weights
2. Ensure a test_set folder exists (create and pre‑fill with a few val images if necessary)
3. Re‑save test images to catch hidden corruptions
4. Run YOLOv8 inference, then apply heuristic post‑processing
5. Save visualised predictions to disk
"""

import os
import shutil
from glob import glob
from ultralytics import YOLO
from yolo_detector.data_utils import reload_and_save_images
from yolo_detector.postprocessing import apply_post_processing_rules
from yolo_detector.inference import draw_detections

def run_inference_and_visualize():
    # --------------------------------------------------------------------------- #
    # 1. Load the trained checkpoint
    # --------------------------------------------------------------------------- #
    best_model = YOLO("models/best.pt")   # path to best.pt

    # --------------------------------------------------------------------------- #
    # 2. Prepare test directory
    # --------------------------------------------------------------------------- #
    base_dir = "data"  # Current project structure
    test_dir = os.path.join(base_dir, "test_set")
    val_dir = os.path.join(base_dir, "images/val")
    predictions_dir = "predictions/test_set"
    
    # Create test directory if it doesn't exist
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory {test_dir} does not exist. Creating it.")
        os.makedirs(test_dir, exist_ok=True)
        
        # Copy up to 5 validation images for testing
        val_images = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for i, file in enumerate(val_images[:5]):
            shutil.copy(os.path.join(val_dir, file), test_dir)
            print(f"Copied {file} to test directory")

    # --------------------------------------------------------------------------- #
    # 3. Sanitise test images (repair colour mode / corruption)
    # --------------------------------------------------------------------------- #
    reload_and_save_images(test_dir)

    # --------------------------------------------------------------------------- #
    # 4. Run inference and apply rule‑based clean‑up
    # --------------------------------------------------------------------------- #
    results = best_model.predict(test_dir, save=False)  # Don't save to runs/predict
    processed_results = apply_post_processing_rules(results)

    # --------------------------------------------------------------------------- #
    # 5. Save the processed detections as visualisations
    # --------------------------------------------------------------------------- #
    os.makedirs(predictions_dir, exist_ok=True)
    
    for result, detections in zip(results, processed_results):
        img_path = result.path
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(predictions_dir, f"processed_{base_name}.jpg")
        
        # Print detected classes for debugging
        print(f"\nProcessing {base_name}:")
        for det in detections:
            class_idx = det.get('class', 0)
            conf = det.get('confidence', 0)
            print(f"  Class {class_idx} with confidence {conf:.2f}")
        
        # Draw detections using our custom function
        draw_detections(img_path, detections, output_path)
        print(f"Saved processed image to: {output_path}")

    print("\nInference complete with post-processing rules applied")

if __name__ == "__main__":
    run_inference_and_visualize() 