# Manga Chat Bubble Detection with YOLOv8

This project focuses on detecting chat bubbles in manga pages using YOLOv8 (You Only Look Once version 8), a state-of-the-art object detection model. The goal is to accurately identify and classify three key elements in manga pages: chat bubbles, narration boxes, and UI elements.

## Project Overview

The project consists of dataset annotation, model training, and evaluation using YOLOv8, an object detection algorithm that is both fast and highly accurate. I personally annotated the dataset using Label Studio and then trained the model on the annotated data to detect three categories: **chat bubbles**, **narration boxes**, and **UI elements**.

### Key Features:
- **Dataset Annotation:** Annotated the dataset manually using Label Studio with three distinct labels: `bubble`, `narration`, and `UI`.
- **YOLOv8 for Object Detection:** Trained the YOLOv8 model to detect the annotated objects with high accuracy.
- **Custom Data Preprocessing:** Handled image truncation and preprocessing to optimize the dataset for model training.
- **Model Evaluation:** Evaluated the model's performance and fine-tuned it to improve accuracy.

## Results

The trained YOLOv8 model achieved promising results in detecting and classifying the three elements in manga pages: **chat bubbles**, **narration boxes**, and **UI elements**.

- **Accuracy:** The model demonstrated strong accuracy in correctly classifying objects, with an average precision and recall score of [insert scores if available].
- **Model Evaluation:** The evaluation on a separate test set showed the model's ability to generalize well, performing with minimal false positives and false negatives.
- **Challenges Overcome:** Some challenges included handling images with truncated or overlapping elements, which were addressed by optimizing the dataset and preprocessing steps.

This project not only showcased my ability to work with image data but also gave me hands-on experience in fine-tuning and evaluating object detection models.

### Future Improvements:
- Fine-tuning the model on more diverse manga pages.
- Exploring multi-label detection to handle complex cases where multiple objects overlap.

## Technologies Used:
- **YOLOv8** for object detection
- **Label Studio** for dataset annotation
- **PyTorch** for model training and evaluation (Note: PyTorch was used instead of TensorFlow)
- **PIL (Python Imaging Library)** for image preprocessing

## How to Use:
1. **Clone the repository** to your local machine:
   ```bash
   git clone https://github.com/ebhon/YOLO-manga-bubble-detector.git

