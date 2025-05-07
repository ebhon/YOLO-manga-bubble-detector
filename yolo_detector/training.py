"""
Functions for training the YOLO model.
"""

import os
import yaml
from collections import Counter
from ultralytics import YOLO

def compute_class_weights(class_counts: Counter) -> dict:
    """
    Compute class weights based on class distribution.
    
    Args:
        class_counts: Counter object containing class counts
        
    Returns:
        Dictionary mapping class IDs to their weights
    """
    total = sum(class_counts.values())
    return {class_id: total / (len(class_counts) * count) 
            for class_id, count in class_counts.items()}

def write_data_yaml(base_dir: str, class_weights: dict) -> str:
    """
    Write the data YAML file for YOLO training.
    
    Args:
        base_dir: Base directory containing the dataset
        class_weights: Dictionary of class weights
        
    Returns:
        Path to the created YAML file
    """
    data_yaml = {
        'path': os.path.abspath(base_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: "bubble",
            1: "narration",
            2: "other",
            3: "text",
            4: "ui"
        },
        'weights': class_weights
    }
    
    yaml_path = os.path.join(base_dir, 'data_balanced.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    return yaml_path

def train_model(run_name: str, data_yaml_path: str) -> str:
    """
    Train a YOLOv8 model on the prepared dataset.
    
    Args:
        run_name: Name of this training run
        data_yaml_path: Path to the data YAML file
        
    Returns:
        Path to the best model weights
    """
    # Initialize model
    model = YOLO('yolov8n.pt')
    
    # Training configuration
    training_args = {
        'data': data_yaml_path,
        'epochs': 50,
        'imgsz': 512,
        'patience': 15,
        'batch': 2,  # Reduced batch size
        'cos_lr': True,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'degrees': 10.0,
        'scale': 0.5,
        'workers': 0,  # Reduced workers
        'project': 'models',  # Save directly in models directory
        'name': run_name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'Adam',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'cache': False,  # Disable caching
        'amp': False,    # Disable mixed precision
        'device': 0     # Force GPU
    }
    
    # Train the model
    results = model.train(**training_args)
    
    # Get the path to the best weights
    best_weights_path = os.path.join('models', run_name, 'weights', 'best.pt')
    
    return best_weights_path