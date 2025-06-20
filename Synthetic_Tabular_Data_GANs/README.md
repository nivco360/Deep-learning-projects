# Synthetic Tabular Data Generation with GAN and cGAN

This project explores the use of Generative Adversarial Networks (GAN) and Conditional GANs (cGAN) to generate high-quality synthetic tabular data. The study replicates and evaluates real-world statistical and machine learning properties of the Adult Income dataset. Our goal was to assess the realism of generated data through statistical similarity and predictive performance.

---

## 📅 Overview

Tabular data generation poses unique challenges due to mixed feature types, sparsity, and statistical interdependencies. We compare GAN and cGAN approaches:

- **GAN**: Trains to mimic overall data distribution
- **cGAN**: Receives auxiliary information (e.g., class label) to condition the generation process

The project evaluates how well each method recreates the marginal distributions and preserves machine learning utility.

---

## 📊 Dataset Description

**Adult Income Dataset (UCI Repository)**

- **Binary classification task**: Predict income ≤\$50K or >\$50K
- **Features**:
  - 6 continuous (e.g., age, hours-per-week)
  - 8 categorical (e.g., workclass, education, occupation)
- Total: 14 attributes + label
- Imbalanced dataset: \~76% earn ≤\$50K

---

## 🔄 Preprocessing Steps

- Label encoding for categorical features
- Normalization for continuous features
- No class rebalancing was applied
- All features were retained
- Final shape: Encoded categorical + normalized continuous features

---

## 🛠️ Model Architectures

### ✨ Generator (for both GAN and cGAN):

- Input: Noise vector (dim=500) [+ one-hot label vector in cGAN]
- Layers:
  - Linear → BatchNorm → LeakyReLU → Dropout(0.3)
  - Linear → BatchNorm → LeakyReLU → Dropout(0.3)
  - Linear → BatchNorm → LeakyReLU
  - Output: Linear → match data feature dim

### 🕵️ Discriminator:

- Input: Data sample [+ one-hot label vector in cGAN]
- Layers:
  - Linear → LeakyReLU → Dropout(0.4)
  - Linear → LeakyReLU → Dropout(0.4)
  - Output: Sigmoid

---

## 💪 Training Configuration

- Optimizer: Adam (lr=0.0002, betas=(0.5, 0.999)) or RMSProp
- Epochs: up to 1000 with early stopping
- Batch sizes: [32, 64]
- Loss: BCEWithLogitsLoss (Sigmoid + Binary Cross Entropy)
- Multiple random seeds and grid-search over hyperparameters

### 📊 Figures


- `generator_loss_curve.png`
- `discriminator_loss_curve.png`
- Histograms for selected features
---

### 🔍 Efficacy Evaluation via Classification

Trained RF classifier on:

1. Real data
2. Synthetic data (GAN or cGAN)

Evaluated on real test set (AUC):

| Model | AUC on Real | AUC on Synthetic | Efficacy Ratio |
| ----- | ----------- | ---------------- | -------------- |
| GAN   | 0.90507     | 0.52266          | 0.58888        |
| cGAN  | 0.90507     | 0.5121           | 0.5683         |

## 💭 Discussion & Key Findings

- Both GAN and cGAN failed to produce synthetic data that closely resembled real data (Detection AUC = 1.00)
- cGAN showed greater sensitivity to random seed, reflecting training instability
- Efficacy ratios were low across both models, indicating synthetic data did not serve as an effective substitute for real data
- GAN preserved statistical characteristics better; cGAN was more unstable

---