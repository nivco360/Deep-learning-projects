# 🧠 Deep Learning Projects – Niv Cohen

This repository showcases a collection of deep learning projects developed during my M.A. in Computational Neuroscience at Ben-Gurion University. Each project addresses a different modality and modeling challenge—ranging from image classification and face verification to language generation and synthetic data modeling.

The work reflects a balance between theoretical foundations and practical implementation, with an emphasis on reproducibility, performance analysis, and scientific reporting.

---

## 📁 Projects Overview

### 🔢 1. Neural Network for MNIST Classification

- **Task**: Classify handwritten digits (0–9)
- **Focus**: Batch Normalization and L2 regularization comparisons
- **Highlights**: Stable convergence tracking with visualization of loss curves
- [🔗 View Project](./NeuralNetwork_MNIST)

---

### 🧍‍♂️ 2. Face Verification with Siamese Network (LFW)

- **Task**: One-shot face verification using image pairs
- **Focus**: Contrastive loss training on embedding distances
- **Highlights**: Three model variants (original, BatchNorm, L2); accuracy & AUC metrics
- [🔗 View Project](./CNN_Siamese_LFW)

---

### 🎵 3. Lyrics Generation with RNNs and MIDI Conditioning

- **Task**: Generate lyrics conditioned on melody and rhythm vectors from MIDI files
- **Focus**: Music-informed language modeling with GRUs
- **Highlights**: Word2Vec integration, feature fusion from MIDI + lyrics, loss curve comparison
- [🔗 View Project](./Lyrics_Generation_MIDI_RNN)

---

### 🧾 4. Synthetic Tabular Data Generation with GAN vs. cGAN

- **Task**: Model and reproduce statistical properties of structured tabular data
- **Focus**: Evaluate data realism using classifier-based detection and AUC comparisons
- **Highlights**: Multi-run analysis, histograms of real vs. synthetic distributions, efficacy ratios
- [🔗 View Project](./Synthetic_Tabular_Data_GANs)

---


## 🗂️ Repository Structure

```
Deep-learning-projects/
├── NeuralNetwork_MNIST/
├── CNN_Siamese_LFW/
├── Lyrics_Generation_MIDI_RNN/
├── Synthetic_Tabular_Data_GANs/
└── README.md
```
