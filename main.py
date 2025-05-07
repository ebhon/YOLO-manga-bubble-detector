"""
Main script for manga bubble detection.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent))

from scripts.train import main as train_main
from scripts.prepare_dataset import main as prepare_main

def main():
    parser = argparse.ArgumentParser(description='Manga Bubble Detector')
    parser.add_argument('--mode', type=str, required=True, choices=['prepare', 'train'],
                      help='Mode to run: prepare (dataset) or train (model)')
    parser.add_argument('--input', type=str, default='data/raw',
                      help='Input directory for prepare mode (default: data/raw)')
    parser.add_argument('--output', type=str, default='data',
                      help='Output directory for prepare mode (default: data)')
    
    args = parser.parse_args()
    
    if args.mode == 'prepare':
        prepare_main(args.input, args.output)
    elif args.mode == 'train':
        train_main()

if __name__ == "__main__":
    main()