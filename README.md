# ECoG Hand Gesture Recognition

This project implements a machine learning model for classifying hand gestures using Electrocorticography (ECoG) data. The model is designed to recognize gestures like fist, peace, and open hand from ECoG signals, leveraging advanced signal processing techniques and machine learning algorithms like Common Spatial Patterns (CSP) and Riemannian geometry classifiers.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)

## Overview
Hand gesture recognition using brain signals recorded through ECoG is a challenging task that combines biosignal processing and machine learning techniques. This project uses data from the ECoG_Handpose dataset and applies filters, feature extraction, and classification models to distinguish between different gestures.

Key features of this project:
- ECoG Signal Processing: ECoG signals are filtered and processed to remove noise, such as powerline interference, and emphasize relevant frequency bands (e.g., high gamma range).
- CSP & Riemannian Geometry: Common Spatial Patterns (CSP) are used for feature extraction, while Riemannian geometry-based classifiers are employed to improve accuracy.
- Gesture Classification: Hand gestures such as fist, peace, and open hand are classified using the processed ECoG signals.

## Dataset

The ECoG Handpose dataset can be downloaded from the following link:

ECoG_Handpose Dataset: https://drive.google.com/file/d/1GRnZVuIRp7b3y3Ngvnm1DSLLFV_A1ZiI/view?usp=sharing

The dataset includes:
- Raw ECoG signals recorded from multiple channels.
- Positional data for electrode placement.
- Labels representing hand gestures.

## Installation

To get started with this project, you need to set up a Python environment with the required dependencies.

### Prerequisites

Ensure you have Python 3.7 or later installed. Follow these steps to install the necessary packages:

1. Clone the repository:
```bash
   git clone https://github.com/your-username/ECoG-Handpose.git
   cd ECoG-Handpose
```
2. Install the required dependencies:
```bash
   pip install -r requirements.txt
```
## Data Preprocessing

The ECoG data undergoes several preprocessing steps to enhance the signal and prepare it for classification:
- Filtering: A band-pass filter is applied in the high gamma range (50-300 Hz) to focus on relevant brain activity frequencies.
- Notch Filtering: A notch filter is applied to remove powerline noise at 50 Hz and its harmonics.
- Epoch Creation: ECoG signals are segmented into epochs around gesture events for training the classification model.

## Model Architecture

The model uses a combination of signal processing and machine learning techniques:
- Common Spatial Patterns (CSP): CSP is applied to the filtered ECoG data to extract features that maximize the variance between gestures.
- Riemannian Geometry Classifiers: The extracted features are classified using a Riemannian geometry-based classifier, which is effective for handling covariance matrices derived from the ECoG data.

## Training and Evaluation

The model is trained using cross-validation to ensure robustness. Key aspects of the training and evaluation process include:

- ShuffleSplit Cross-Validation: A ShuffleSplit cross-validator is used to split the data into training and testing sets multiple times, providing a more robust evaluation.
- Classification Accuracy: The performance of the model is evaluated based on the accuracy of classifying different hand gestures.

### Example Training Command:

python script.py

This command will run the script to preprocess the data and classify hand gestures based on the processed ECoG signals.

## Results

The model is evaluated on several metrics to assess its performance:

- Classification Accuracy: Accuracy is computed as the percentage of correctly classified hand gestures across different validation folds.
- Confusion Matrix: A confusion matrix is generated to visualize the classification performance across different gestures.
- Overall Accuracy: The overall accuracy of the model is reported after averaging across cross-validation folds.
