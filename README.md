# Discourse Markers Scripts

A collection of scripts and notebooks for analyzing, visualizing, and modeling discourse markers in linguistic data.

## Overview

This repository contains various tools for working with discourse markers, which are words or phrases that help connect, organize, and manage ideas in speech and writing (such as "however," "therefore," "in addition," etc.).

## Repository Contents

### Data Analysis & Visualization

- [DATAVIZ_DM.ipynb](./DATAVIZ_DM.ipynb) - Visualizations for discourse markers data
- [EDA_plus_DESCRIPTIVE-STATISTICS_MDs.ipynb](./EDA_plus_DESCRIPTIVE-STATISTICS_MDs.ipynb) - Exploratory data analysis and descriptive statistics
- [CALCULATE_SLOPE_OF_DM.ipynb](./CALCULATE_SLOPE_OF_DM.ipynb) - Analysis of slope patterns in discourse markers

### Feature Selection

- [FEATURE_SELECTION.ipynb](./FEATURE_SELECTION.ipynb) - Jupyter notebook for feature selection
- [FEATURE_SELECTION.R](./FEATURE_SELECTION.R) - R script for feature selection
- [FEATURE_SELECTION_oneVSrest.R](./FEATURE_SELECTION_oneVSrest.R) - One-vs-rest approach for feature selection
- [FEATURE_SELECTION_smote.R](./FEATURE_SELECTION_smote.R) - Feature selection with SMOTE for imbalanced data

### Model Training & Evaluation

- [CLASSIFICATION_MODELS_WITH-CURVES.ipynb](./CLASSIFICATION_MODELS_WITH-CURVES.ipynb) - Classification models with performance curves
- [TRAIN_EVALUATE_BASE-MODEL_DATAGENERATOR_SLIDINGWINDOW_PREDICTIONS_correction-padding.ipynb](./TRAIN_EVALUATE_BASE-MODEL_DATAGENERATOR_SLIDINGWINDOW_PREDICTIONS_correction-padding.ipynb) - Training and evaluation pipeline
- [HYPERPARAMETER_TUNING_GPU.py](./HYPERPARAMETER_TUNING_GPU.py) - Script for hyperparameter tuning on GPU
- [keras_tuner_example.py](./keras_tuner_example.py) - Example of using Keras Tuner

### Viterbi Algorithm Implementation

- [VITERBI.ipynb](./VITERBI.ipynb) - Implementation of the Viterbi algorithm
- [VITERBI_DECISION-MODEL_only-continuous-PDAs.ipynb](./VITERBI_DECISION-MODEL_only-continuous-PDAs.ipynb) - Decision model using Viterbi
- [viterbi.py](./viterbi.py) - Python module for Viterbi algorithm

### Utility Scripts

- [GET-MEASURES_SPYDER_h5py_OLIVER.py](./GET-MEASURES_SPYDER_h5py_OLIVER.py) - Script for extracting measures
- [GET-TEXTGRID_WITHSTRESSED-VOWEL.ipynb](./GET-TEXTGRID_WITHSTRESSED-VOWEL.ipynb) - Processing TextGrid files with stressed vowel information
- [train_loader.py](./train_loader.py) - Utilities for loading training data

## Usage

To use these scripts, clone the repository and run the desired notebook or script. Most notebooks are self-contained with explanations of the process.

```bash
git clone https://github.com/saulo-smendes/discourse_markers_scripts.git
cd discourse_markers_scripts
```

For Python scripts, you may need to install dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow keras h5py
```

For R scripts, the following packages are required:

```r
install.packages(c("caret", "randomForest", "e1071", "DMwR"))
```

## Models

Pre-trained models are included in the repository:
- [cnn_all.h5](./cnn_all.h5)
- [cnn_f0.h5](./cnn_f0.h5)
- [cnn_mfcc.h5](./cnn_mfcc.h5)