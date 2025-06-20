# Lyrics Generation with MIDI-Conditioned RNN

This project explores the generation of song lyrics using recurrent neural networks (RNNs) conditioned on MIDI-derived musical features. The model aims to incorporate melodic and rhythmic structure into the language modeling process.

---

## 📚 Overview

The task is framed as a language modeling problem: given a seed text and musical conditioning (melody and rhythm), the model learns to predict the next word in a sequence. Two model variants were tested:

1. **Melody-only conditioning**
2. **Melody + Rhythm conditioning**

Although the generated lyrics were limited in quality, the project offers insights into how symbolic musical data can influence natural language generation.

---

## 🗾️ Dataset and Preprocessing

- **Text Source**: Lyrics corpus
- **Music Source**: Corresponding MIDI files for each song

### 🔄 Preprocessing Pipeline

- MIDI files were parsed to extract:
  - Melody vector
  - Rhythm vector
- Word-level tokenization
- Word2Vec embedding of lyrics
- Conditioning vectors were synchronized with lyric positions

Code for this step is located in `lyrics_midi_preprocess.py`

---

## 🛍️ Architecture

The model integrates textual and musical input through the following components:

- Word2Vec embedding for lyrics
- Melody and rhythm vectors as parallel inputs
- **Three GRU layers**
- **Three Linear layers** with intermediate activation and normalization:
  - ReLU
  - Dropout
  - LayerNorm
- Softmax output layer (predicts word probability distribution)

---

## 🔧 Training Details

- **Optimizer**: SGD
- **Loss**: Cross-Entropy
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Embedding Size**: 300 (Word2Vec)
- **Epochs**: 10

Training was conducted on both model variants using identical hyperparameters.

---

## 📊 Results

### 🔢 Training and Validation Loss

Loss curves show good convergence for both models:



- Melody-only and Melody+Rhythm models both reached stable loss after 10 epochs
- Rhythm input did not improve loss but did not hinder learning either

---

## ⚠️ Limitations

- Lyrics generated were **semantically poor** and lacked structure
- Word2Vec did not capture sufficient context for coherence
- Future improvements may include:
  - Transformer-based decoding
  - Attention over musical context
  - Filtering unaligned or noisy sequences

---

## 📄 How to Run

1. Preprocess MIDI and lyrics:

```bash
python lyrics_midi_preprocess.py
```

2. Train the model:

```bash
python lyrics_rnn_main.py
```

Model and training code are in `lyrics_rnn_model.py`

---

## 📙 Author

Niv Cohen

