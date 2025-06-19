# CNN Siamese Network for One-Shot Face Verification on LFW Dataset

This project implements and evaluates multiple Siamese Neural Network architectures for one-shot face recognition using the Labeled Faces in the Wild (LFW) dataset. The goal was to identify whether two face images belong to the same individual.

## 🧠 Overview

The study compares the original architecture from the reference paper with two new architectures we designed. All models were trained using binary classification (same vs. different person) and evaluated using validation and test accuracy.

## 🛠️ Methodology

### 🗾️ Dataset

- **LFW dataset** with 13,233 face images of 5,749 people.
- Train/Validation/Test split:
  - Train: 1,760 pairs (50% positive/negative)
  - Validation: 440 pairs
  - Test: 1,000 pairs (balanced)

### 🛠️ Preprocessing

- Images resized to **105×105** pixels (as in the original paper).
- Normalization: mean subtraction and division by standard deviation.

### 🗪 Training Setup

- Loss: Binary Cross Entropy
- Optimizer: Adam
- Batch Sizes Tested: 32, 64, 128
- Learning Rates Tested: 1e-3, 1e-4, 1e-5
- Early stopping with `patience=5`, `tolerance=0.001`
- Regularization: L2 with λ = 1e-5
- Weight initialization: Xavier & Normal distribution

### 🧬 Architectures

Three Siamese CNN architectures were tested:

1. **Original (from literature)**
2. **new\_arch\_1** – small filters, fewer FC units (2048→1024), AdaptiveAvgPool
3. **new\_arch\_2** – same as above + BatchNorm

## 📊 Results

- Best model: `new_arch_1`
- **Test Accuracy:** 66.9%
- **Validation Accuracy:** 61.5%
- Training showed stable convergence with clear improvement in both `train_accuracy` and `train_loss`.

## 📈 Key Findings

- All models showed similar accuracy (53%–67%).
- Smaller filters and fewer parameters improved training stability.
- BatchNorm helped early convergence but didn’t significantly boost final accuracy.
- Best trade-off between complexity and performance achieved with `new_arch_1`.

## 🔍 Confusion Matrix Insight

- Balanced performance on both positive and negative test pairs.
- Some false positives occurred due to facial similarity in pose or features.

