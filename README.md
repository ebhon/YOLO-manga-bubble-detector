# Manga Bubble Detector

A YOLOv8-based detector for manga speech bubbles and text boxes. This project uses computer vision and deep learning to automatically detect and classify different types of text elements in manga pages, making it easier to process manga for translation, analysis, or digital enhancement.

## Features

- 🔍 Detects 5 types of manga text elements:
  - Speech bubbles
  - Narration boxes
  - Other text containers
  - Text content
  - UI elements
- 🎯 High accuracy with YOLOv8 architecture
- 🛠️ Built-in dataset preparation tools
- 📊 Automatic train/val split with class balancing
- 🔄 Image corruption detection and repair
- 🎨 Visualization tools for predictions
- ⚡ Fast inference with GPU support

## Technologies Used

- **Deep Learning Framework**: PyTorch
- **Object Detection**: YOLOv8
- **Image Processing**: OpenCV, Pillow
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Development**: Python 3.8+

## Project Structure

```
.
├── data/                    # Dataset directory
│   ├── raw/                # Raw data
│   │   ├── raw_images/     # Original manga images
│   │   └── raw_labels/     # YOLO format labels
│   ├── images/             # Processed images
│   │   ├── train/         # Training images
│   │   └── val/           # Validation images
│   └── labels/             # Processed labels
│       ├── train/         # Training labels
│       └── val/           # Validation labels
├── models/                 # Trained models
├── predictions/            # Inference results
├── scripts/               # Utility scripts
└── yolo_detector/         # Core detector code
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/manga-bubble-detector.git
cd manga-bubble-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

Place your raw manga images and their corresponding YOLO format labels in:
- `data/raw/raw_images/` - for images
- `data/raw/raw_labels/` - for labels

Then run:
```bash
python main.py --mode prepare
```

This will:
- Split the dataset into train/val sets
- Fix any corrupted images
- Save the processed dataset in `data/images/` and `data/labels/`

### 2. Train Model

To train the model:
```bash
python main.py --mode train
```

This will:
- Train a YOLOv8 model on the prepared dataset
- Save the best model in `models/best.pt`
- Save training artifacts in `models/run{N}/`

### 3. Run Inference

To run inference on test images:
```bash
python scripts/infer.py
```

Or with custom paths:
```bash
python scripts/infer.py --model models/best.pt --test_dir data/test_set
```

### 4. Visualize Predictions

To visualize the predictions:
```bash
python scripts/visualize_predictions.py
```

This will:
- Run inference on test images
- Apply post-processing rules
- Save visualized predictions in `predictions/test_set/`

## Advanced Usage

### Custom Dataset Preparation

If your data is in a different location:
```bash
python main.py --mode prepare --input path/to/raw/data --output path/to/output
```

### Custom Training

The training script uses these default parameters:
- Model: YOLOv8n
- Image size: 512x512
- Batch size: 2
- Epochs: 50
- Patience: 15

You can modify these in `yolo_detector/training.py`.

## Class Labels

The detector recognizes these classes:
- 0: bubble (speech bubble)
- 1: narration (narrator's text box)
- 2: other (other text containers)
- 3: text (text content)
- 4: ui (user interface elements)

## Performance

The model achieves the following performance on our validation set (115 images, 1609 instances):

| Class      | Precision | Recall  | mAP50   | mAP50-95  |
|------------|-----------|---------|---------|-----------|
| All        | 0.959     | 0.826   | 0.900   | 0.750     |
| Bubble     | 0.981     | 0.951   | 0.984   | 0.855     |
| Narration  | 0.936     | 0.942   | 0.988   | 0.877     |
| Text       | 0.994     | 0.911   | 0.970   | 0.775     |
| UI         | 0.926     | 0.500   | 0.658   | 0.493     |

**Inference Speed**: 1.7ms preprocess, 19.0ms inference, 0.0ms loss, 1.9ms postprocess per image

**Model Size**: 72 layers, 3,006,623 parameters, 8.1 GFLOPs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.