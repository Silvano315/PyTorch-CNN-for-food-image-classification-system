# PyTorch-CNN-for-food-image-classification-system

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methods with PyTorch](#methods-with-pytorch)
- [Visualization and Preprocessing](#visualization-and-preprocessing)
- [Baseline CNN Models](#custom-cnn-models)
  - [Model Architecture](#model-architecture)
  - [Results of Baseline CNNs](#results-of-custom-cnns)
- [Transfer Learning](#transfer-learning)
  - [Model Architecture](#transfer-learning-model-architecture)
  - [GRAD-CAM implementation](#grad-cam-implementation)
  - [Results of Transfer Learning](#results-of-transfer-learning)
- [Key Insights](#key-insights)
- [Conclusions](#conclusions)
- [How to Run](#how-to-run)
- [EXTRA: GUI for a user-friendly system](#EXTRA-GUI-for-a-user-friendly-system)


## Project Overview

This repository is the sixth project of the master's degree in AI Engineering with [Profession AI](https://profession.ai), all the credits for the requests and idea go to this team. 

GourmetAI Inc., a renowned food technology company, faces increasing challenges in improving the accuracy and efficiency of food image classification systems. GourmetAI Inc. requested the development of an advanced food image classification model using deep learning techniques.

Project Benefits:
- Improved User Experience
- Optimized Business Processes
- Technology Innovation

Project Objectives:
- Augmentation Strategies: implement different augmentation techniques to enrich the dataset, improving data variability and quality.
- Network Architectures and Transfer Learning: select and implement one or more neural network architectures suitable for the problem, using transfer learning to exploit pre-trained models.
- Fine Tuning and Hyperparameters Choice: create a custom classifier, choose the hyperparameters and optimize the model through training and validation processes.
- Validation and Regularization: retraining with validation and regularization techniques to improve the model performance.

## Dataset

The project will be based on the Food Classification dataset, enriched with augmentation techniques to improve the diversity and quality of the available data. This is the [link](https://proai-datasets.s3.eu-west-3.amazonaws.com/dataset_food_classification.zip) to download the dataset.

This dataset is composed by:
- 14 classes with
  - 640 images for training
  - 160 images for validation
  - 200 images for test

## 🛠️ Methods with PyTorch

This project leverages PyTorch to implement a robust and flexible system for training and evaluating CNN models. I've designed several custom classes to streamline the experimental process, you can find them [here](src/models.py)

### Experiment Class

The `Experiment` class serves as the backbone of the training pipeline. It manages:

- Logging of training progress
- Saving and loading of model weights
- Visualization of training history
- Exporting of results

Key features:
- Automatic creation of directory structure for each experiment
- CSV logging of training and validation metrics
- Plotting of training history
- JSON export of final results

### Callback System

I've implemented a callback system inspired by Keras, allowing for flexible control of the training process:

1. **EarlyStopping**: Prevents overfitting by stopping training when a monitored metric has stopped improving.
2. **ModelCheckpoint**: Saves the best model based on a specified metric.
3. **ReduceLROnPlateau**: Reduces learning rate when a metric has stopped improving.

### Model Architecture

I've implemented two main model architectures:

1. **BaselineCNN**: A simple CNN architecture for baseline comparisons.
2. **EfficientNetTransfer**: A transfer learning model based on EfficientNet, allowing for easy experimentation with different EfficientNet versions.

### Training and Evaluation Functions

The `train_model` function encapsulates the entire training loop, including:

- Epoch-wise training and validation
- Logging of metrics
- Execution of callbacks
- Resuming training from checkpoints

The `validate` and `get_predictions` functions provide easy-to-use interfaces for model evaluation and inference.

### Grad-CAM Visualization

I've implemented the Grad-CAM algorithm in the `apply_gradcam` function, allowing for visual explanation of model decisions.

This modular and extensible design allows for easy experimentation with different models, training strategies, and visualization techniques.

## Methods with PyTorch

## Visualization and Preprocess

## Baseline CNN Model

### Model Architecture

### Results of Baseline CNNs

## Transfer Learning

### Model Architecture

### GRAD-CAM implementation

### Results of Transfer Learning

## Key Insights

## Conclusions

## How to Run

## EXTRA: GUI for a user-friendly system

