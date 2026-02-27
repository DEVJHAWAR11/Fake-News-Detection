# Fake News Detection 📰

> Deep learning pipeline for classifying news articles as **real or fake** using Bidirectional LSTM and pretrained GloVe word embeddings.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Fake%20News-yellow)](https://www.kaggle.com/c/fake-news)

---

## Overview

Fake news spreads faster than corrections. This project builds a deep learning classifier trained on 20,000+ labeled news articles to automatically detect misinformation.

The core idea: instead of manually engineering features, I let the model learn what "fake" language looks like directly from raw text — using word embeddings to capture semantic meaning and an LSTM to capture sequential context across sentences.

---

## Models Compared

| Model | Approach | Accuracy |
|---|---|---|
| Naive Bayes | TF-IDF baseline | ~88% |
| LSTM | Word embeddings + sequential model | ~97% |
| **Bidirectional LSTM** | Reads text forward + backward | **~99%** |
| GRU | Gated recurrent unit variant | ~98% |
| CNN | 1D convolution over token sequences | ~97% |

Bidirectional LSTM gave the best results — reading each article in both directions captures richer context than a single-direction LSTM.

---

## Architecture

```
Raw Text
    │
    ▼
Tokenization + Padding (max_len = 500)
    │
    ▼
Embedding Layer (GloVe 100d pretrained vectors)
    │
    ▼
Bidirectional LSTM (128 units)
    │
    ▼
Dropout (0.5) + Dense (64, ReLU)
    │
    ▼
Dense (1, Sigmoid) → Real / Fake
```

**Why GloVe?** Pretrained on 6B tokens — the model starts with real-world word relationships instead of learning from scratch. "president" and "election" are already close in the embedding space before training even begins.

**Why Bidirectional?** A standard LSTM reads left-to-right. BiLSTM reads both ways — so "not good" and "good" aren't treated the same even though "good" appears in both.

---

## Project Structure

```
Fake-News-Detection/
├── Fake_news_NLP.ipynb      # Full training notebook — EDA, preprocessing, model training, evaluation
├── app/                     # Streamlit web app
├── model/                   # Saved trained model weights
├── images/                  # Confusion matrix, loss curves, word clouds
└── requirements.txt
```

---

## Key Results

- **Accuracy:** 99%+ on test set
- **Precision / Recall / F1:** All above 0.98
- Confusion matrix and training curves in `/images`
- Baseline (Naive Bayes TF-IDF): 88% → Deep learning improvement: **+11%**

---

## Run Locally

```bash
git clone https://github.com/DEVJHAWAR11/Fake-News-Detection.git
cd Fake-News-Detection
pip install -r requirements.txt
streamlit run app/app.py
```

**Dataset:** Download from [Kaggle Fake News Competition](https://www.kaggle.com/c/fake-news/data) and place in root directory.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | Keras + TensorFlow |
| Word Embeddings | GloVe (100-dimensional, 6B tokens) |
| Model Architecture | Bidirectional LSTM |
| Data Processing | Pandas, NumPy, NLTK |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Web App | Streamlit |

---

## Author

**Dev Jhawar** — [GitHub](https://github.com/DEVJHAWAR11) | [LinkedIn](https://linkedin.com/in/dev-jhawar11) | KIIT University, CSE
