"""
Script to run inference on test images using the trained YOLO model.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from yolo_detector.inference import run_folder_inference

def main():
    parser = argparse.ArgumentParser(description='Run inference on test images')
    parser.add_argument('--model', type=str, default='models/best.pt',
                      help='Path to the model weights (default: models/best.pt)')
    parser.add_argument('--test_dir', type=str, default='data/test_set',
                      help='Path to test images directory (default: data/test_set)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    
    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"Error: Test set directory not found at {args.test_dir}")
        sys.exit(1)
    
    print(f"Running inference on {args.test_dir} using model {args.model}...")
    run_folder_inference(args.model, args.test_dir)

if __name__ == "__main__":
    main()