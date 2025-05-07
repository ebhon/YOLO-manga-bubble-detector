"""
Script to train the YOLO model on the prepared dataset.
"""

import os
import sys
from pathlib import Path
from collections import Counter
import shutil

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from yolo_detector.data_utils import count_classes_in_label_file
from yolo_detector.training import compute_class_weights, write_data_yaml, train_model

def get_next_run_name(models_dir: str) -> str:
    """
    Get the next available run name by checking existing run directories.
    
    Args:
        models_dir: Path to the models directory
        
    Returns:
        Next available run name (e.g., 'run1', 'run2', etc.)
    """
    existing_runs = [d for d in os.listdir(models_dir) if d.startswith('run')]
    if not existing_runs:
        return 'run1'
    
    # Extract run numbers and find the highest
    run_numbers = []
    for run in existing_runs:
        try:
            num = int(run[3:])  # Extract number after 'run'
            run_numbers.append(num)
        except ValueError:
            continue
    
    next_num = max(run_numbers) + 1 if run_numbers else 1
    return f'run{next_num}'

def main():
    # Configuration
    base_dir = "data"
    labels_dir = os.path.join(base_dir, "labels")
    models_dir = "models"
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Get next run name
    run_name = get_next_run_name(models_dir)
    print(f"Starting training run: {run_name}")
    
    # Compute class weights from training labels
    train_labels_dir = os.path.join(labels_dir, "train")
    class_counts = Counter()
    for label_file in os.listdir(train_labels_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(train_labels_dir, label_file)
            class_counts.update(count_classes_in_label_file(label_path))
    
    class_weights = compute_class_weights(class_counts)
    
    # Write data YAML
    data_yaml_path = write_data_yaml(base_dir, class_weights)
    
    # Train model
    print("Training model...")
    best_weights_path = train_model(run_name, data_yaml_path)
    
    # Copy the best weights to models directory
    best_weights_name = os.path.basename(best_weights_path)
    final_weights_path = os.path.join(models_dir, best_weights_name)
    shutil.copy2(best_weights_path, final_weights_path)
    print(f"Best model copied to: {final_weights_path}")

if __name__ == "__main__":
    main()