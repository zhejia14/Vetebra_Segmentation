# Vertebra Segmentation using Mask R-CNN

This repository contains code for vertebral segmentation in medical images using Mask R-CNN with 3-fold cross-validation. The model detects and segments individual vertebrae and evaluates performance using Dice coefficient.

## Project Overview

This project implements an instance segmentation model for vertebrae in medical images. The model is based on Mask R-CNN architecture with a ResNet-50 backbone pre-trained on COCO dataset and fine-tuned for vertebra segmentation.

### Key Features:
- Vertebra instance segmentation using Mask R-CNN
- 3-fold cross-validation for model evaluation
- Evaluation using Dice coefficient
- Visualization of segmentation results with vertebra contours and centroids

## Dataset Structure

The dataset should be organized in the following structure:
```
dataset/
├── f01/
│   ├── image/
│   │   └── (image files)
│   └── label/
│       └── (mask files)
├── f02/
│   ├── image/
│   │   └── (image files)
│   └── label/
│       └── (mask files)
└── f03/
    ├── image/
    │   └── (image files)
    └── label/
        └── (mask files)
```

Each fold contains vertebra images in the `image` directory and corresponding segmentation masks in the `label` directory.

## Model Architecture

The model is based on Mask R-CNN with the following components:
- ResNet-50 backbone with Feature Pyramid Network (FPN)
- Region Proposal Network (RPN)
- Region of Interest (RoI) alignment
- Bounding box regression and classification heads
- Mask prediction head

## Installation Requirements

```
torch
torchvision
opencv-python (cv2)
numpy
matplotlib
PIL
```

## Usage

### Training

To train the model:

```bash
python train.py [save_path]
```

Where:
- `save_path`: Directory to save the trained model checkpoints

The training process:
- Uses 3-fold cross-validation (2 folds for training, 1 for validation)
- Runs for 150 epochs
- Saves checkpoints every 25 epochs
- Uses SGD optimizer with momentum
- Evaluates on validation set after each epoch

### Evaluation

To evaluate a trained model:

```bash
python eval.py [model_path] [test_dataset_path] [threshold]
```

Where:
- `model_path`: Path to the trained model checkpoint
- `test_dataset_path`: Path to the test dataset directory
- `threshold`: Confidence threshold for mask prediction (e.g., 0.5)

The evaluation script:
- Loads the specified model
- Processes each image in the test dataset
- Computes Dice coefficient for each vertebra
- Visualizes results with contours and centroids
- Saves output images and segmentation masks

## Implementation Details

### Files Description

- `Dataset.py`: Custom dataset class for loading and processing vertebra images and masks
- `Maskrcnn_Model.py`: Model architecture definition
- `train.py`: Training script with cross-validation setup
- `eval.py`: Evaluation script with visualization
- `Evaluation.py`: Additional evaluation utilities

### Performance Metrics

The model performance is evaluated using:
- Dice coefficient (DC): Measures overlap between predicted and ground truth segmentations
- Average Dice score across all vertebrae

## Results Visualization

The evaluation script generates visualization with three panels:
1. Original image
2. Vertebra location (centroids)
3. Segmentation contours overlaid on the original image

## Example Results

Upon successful evaluation, the script will output:
- Dice coefficient for each vertebra
- Average Dice coefficient across all vertebrae
- Visualization of segmentation results
- Saved segmentation masks

## License

**TBD**
