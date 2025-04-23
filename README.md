# Manga Chat Bubble Detection with YOLOv8

This project focuses on detecting chat bubbles in manga pages using YOLOv8 (You Only Look Once version 8), a state-of-the-art object detection model. The goal is to accurately identify and classify key elements in manga pages: chat bubbles, narration boxes, UI elements, and text.

## Project Overview

The project consists of dataset annotation, model training, and evaluation using YOLOv8, an object detection algorithm that is both fast and highly accurate. I personally annotated the dataset using Label Studio and then trained the model on the annotated data to detect four categories: **chat bubbles**, **narration boxes**, **UI elements**, and **text**.

### Key Features:
- **Dataset Annotation:** Annotated the dataset manually using Label Studio with four distinct labels: `bubble`, `narration`, `UI`, and `text`. Special attention was given to separating boxes for conjoined narration boxes.
- **YOLOv8 for Object Detection:** Trained the YOLOv8 model to detect the annotated objects with high accuracy.
- **Active Learning Loop:** Implemented an active learning approach by training the model, predicting on new data, validating with Label Studio, and iteratively refining the dataset through retraining.
- **Custom Data Preprocessing:** Handled image truncation and preprocessing to optimize the dataset for model training.
- **Model Evaluation:** Evaluated the model's performance and fine-tuned it to improve accuracy.

## Results

The trained YOLOv8 model achieved promising results in detecting and classifying the four elements in manga pages: **chat bubbles**, **narration boxes**, **UI elements**, and **text**.

- **Best Metrics:**
  - Best mAP@0.5: 0.9905
  - Best mAP@0.5:95: 0.9034
  - Best Precision: 0.9807
  - Best Recall: 0.9997

This project not only showcased my ability to work with image data but also gave me hands-on experience in fine-tuning and evaluating object detection models.

### Future Improvements:
- Fine-tuning the model on more diverse manga pages.
- Exploring multi-label detection to handle complex cases where multiple objects overlap.
- Continuing to expand the dataset and improve the active learning loop for better performance.

## Technologies Used:
- **YOLOv8** for object detection
- **Label Studio** for dataset annotation
- **PyTorch** for model training and evaluation (Note: PyTorch was used instead of TensorFlow)
- **PIL (Python Imaging Library)** for image preprocessing

## How to Use:
1. **Clone the repository** to your local machine:
   ```bash
   git clone https://github.com/ebhon/YOLO-manga-bubble-detector.git
2. Install the required dependencies:
3. pip install -r requirements.txt
4. Prepare your dataset according to the annotation guidelines.
5. Run the training script to start training the YOLOv8 model.
6. Use the inference script to make predictions on new manga pages.

## Contributing:
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License:
This project is licensed under the MIT License - see the LICENSE file for details.