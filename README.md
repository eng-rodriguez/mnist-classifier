# Convolutional Neural Network for MNIST Classification

## Overview

This repository provides a foundational project for learning and understanding the application of Convolutional Neural Networks (CNNs) in image classification. Specifically, it uses the MNIST dataset, a benchmark dataset of handwritten digits, to demonstrate how CNNs can achieve high accuracy in image recognition tasks.

The project is beginner-friendly and focuses on provided clear, structured code that showcase the essential components of building and training a CNN. 

## Features
- **CNN Architecture**: Implements a simple yet effective CNN model for digit classification.
- **MNIST Dataset Integration**: Automatically downloads and preprocesses the MNIST dataset.
- **Training and Evaluation**: Includes scripts for training the model and evaluating its performance on test data.
- **Visualization**: Provides tools for visualizing training metrics and example predictions.

## Dataset
The MNIST dataset consists of:
- 60,000 training images
- 10,000 test images

Each image is a grayscale 28x28 pixel image of a handwritten digit (0-9).

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.8 or higher
- Torchvision
- Matplolib
- NumPy

You can install the required Python packages using:
```bash
pip install -r requirements.txt
```

## Usage

### Clone the repository

```bash
git clone https://github.com/eng-rodriguez/mnist-classifier.git
cd mnist-classifier
```

### Train the Model

Run the training script to train the CNN on the MNIST dataset:

```bash
python train.py
```

This will output training metrics (e.g., loss, accuracy) and save the trained model in the models/ directory

## Key Components

### Model Architecture

- Convolutional layers for feature extraction.
- Max-pooling layers for down-sampling.
- Fully connected layers for classification.

### Training

- Optimizer: Adam optimizer with learning rate scheduling
- Loss Function: Categorical cross-entropy.
- Metrics: Loss and Accuracy.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
