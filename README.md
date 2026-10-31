# CIFAR-10 Image Classification with a CNN

An end-to-end deep learning project that trains a convolutional neural network from scratch on the CIFAR-10 benchmark (50,000 training and 10,000 test images, 32x32 RGB, 10 classes). The project goes beyond reporting accuracy: it diagnoses overfitting through training vs validation curves, applies Dropout to regularize the model, and uses a confusion matrix to identify which classes the model struggles with (notably cat vs dog).

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626)

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Results](#results)
- [Key Observations](#key-observations)
- [Project Workflow](#project-workflow)
- [How to Run](#how-to-run)
- [Possible Improvements](#possible-improvements)

## Dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) contains 60,000 32x32 colour images in 10 classes (6,000 per class), split into 50,000 training and 10,000 test images. It is loaded directly in the notebook with `keras.datasets.cifar10.load_data()`, so no manual download is needed.

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**Citation:** Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.* University of Toronto.

## Model Architecture
| Layer | Details |
|---|---|
| Input | 32 x 32 x 3 |
| Conv2D + MaxPooling2D | 32 filters, 3x3, ReLU, pool 2x2 |
| Conv2D + MaxPooling2D | 64 filters, 3x3, ReLU, pool 2x2 |
| Conv2D + MaxPooling2D | 128 filters, 3x3, ReLU, pool 2x2 |
| Flatten | |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Output Dense | 10 units, softmax |

## Training Setup
- **Preprocessing:** pixel values scaled to the 0 to 1 range (division by 255). No reshape needed since images are already `(32, 32, 3)`.
- **Optimizer:** Adam
- **Loss:** `sparse_categorical_crossentropy` (labels are integers, not one-hot)
- **Metric:** accuracy
- **Epochs:** 15, **batch size:** 64, **validation split:** 10%

## Results
| Model | Train acc. | Val acc. | Test acc. |
|---|---|---|---|
| Baseline (no Dropout, filters 64-64-128) | 90.5% | 71.5% | 69.4% |
| Final (Dropout 0.5, filters 32-64-128) | XX% | XX% | XX% |

The notebook includes:
- Training vs validation accuracy and loss curves
- A 10x10 confusion matrix (seaborn heatmap with class labels)
- Predictions on 5 random test images, with wrong predictions marked in red

## Key Observations
- **Overfitting:** in the baseline run, training accuracy climbed to about 90% while validation accuracy stalled around 71 to 73%, and validation loss rose from about 0.8 to 1.17. This gap is the clearest sign that the model was memorizing training data. Dropout was added to reduce it.
- **Most confused classes:** cat and dog. In the baseline confusion matrix, 260 cats were predicted as dogs and 101 dogs as cats. These classes share similar shapes, poses and backgrounds at 32x32 resolution.
- **Easiest classes:** vehicles such as automobile, ship and truck, as well as horse, were classified most reliably.

## Project Workflow
1. Import libraries and load CIFAR-10 (the train and test splits are already provided)
2. Inspect shapes (`y_train` is `(50000, 1)`, so labels are flattened for the confusion matrix)
3. Visualize 10 sample images with class names
4. Normalize pixels to 0 to 1
5. Build, compile and train the CNN
6. Plot accuracy and loss curves, then evaluate on the test set
7. Build the confusion matrix and analyze errors
8. Predict on random test images and mark mistakes

## How to Run
1. Open `cifar10_cnn.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime, Change runtime type, T4 GPU** (training is roughly 20x faster than on CPU).
3. Select **Runtime, Run all**.

Requirements: `tensorflow`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (all preinstalled on Colab).

## Possible Improvements
- Data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`)
- `BatchNormalization` after each convolution
- `EarlyStopping` on validation loss
- Deeper architectures or transfer learning (for example, ResNet or MobileNet)
