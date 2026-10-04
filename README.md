# Kaggle NLP with Disaster Tweets — Pocket Data Science V

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://python.org)
[![Platform: Android Termux](https://img.shields.io/badge/Platform-Android%20Termux-black?logo=android&logoColor=white)](https://termux.dev)
[![Agent: Google Antigravity CLI](https://img.shields.io/badge/AI%20Agent-Google%20Antigravity%20CLI-4285F4?logo=google&logoColor=white)](https://antigravity.google)
[![Kaggle Challenge](https://img.shields.io/badge/Kaggle-Natural%20Language%20Processing%20with%20Disaster%20Tweets-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/nlp-getting-started)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, systematic natural language processing study for the [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) challenge. Built, trained, cross-validated, and ensembled entirely on a consumer **Android smartphone CPU** using **Termux PRoot** and **Google Antigravity CLI (`agy`)**.

---

## 🏆 The 5-Tier Experiment Progression

Each experiment establishes a disciplined step up in modeling complexity, evaluated with **5-Fold Stratified Cross-Validation** using Kaggle's official metric: `F1-Score` on real disaster tweets (Class 1).

| Exp | Script | Model / Strategy | 5-Fold OOF F1 | 5-Fold OOF Acc | Train Time | Test Submission |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **01** | [`exp01_baselines.py`](exp01_baselines.py) | Constant Majority Class (All 0s) | 0.0000 | 57.03% | <0.1s | [`submission_majority.csv`](submission_majority.csv) |
| **01b** | [`exp01_baselines.py`](exp01_baselines.py) | Empirical Random Prior (`p=0.430`) | 0.4145 | 49.98% | <0.1s | [`submission_random.csv`](submission_random.csv) |
| **02** | [`exp02_countvec_naive_bayes.py`](exp02_countvec_naive_bayes.py) | CountVectorizer (1,2-gram) + Multinomial Naive Bayes | 0.7509 | 80.13% | 2.4s | [`submission_countvec_nb.csv`](submission_countvec_nb.csv) |
| **03** | [`exp03_tfidf_linear_models.py`](exp03_tfidf_linear_models.py) | Sublinear TF-IDF + Ridge Classifier (`alpha=1.0`) | 0.7581 | 80.30% | 2.2s | [`submission_tfidf_ridge.csv`](submission_tfidf_ridge.csv) |
| **03b** | [`exp03_tfidf_linear_models.py`](exp03_tfidf_linear_models.py) | Sublinear TF-IDF + Logistic Regression (`C=1.5`, `t=0.45`) | 0.7646 | 80.11% | 2.7s | [`submission_tfidf_logistic.csv`](submission_tfidf_logistic.csv) |
| **04** | [`exp04_catboost_text.py`](exp04_catboost_text.py) | CatBoost Native Text Gradient Boosted Trees | 0.7560 | 78.88% | 158.5s | [`submission_catboost.csv`](submission_catboost.csv) |
| **05** | [`exp05_ensemble.py`](exp05_ensemble.py) | **Multi-Model Weighted Ensemble (40% LR + 10% NB + 50% Ridge, `t=0.43`)** | **0.7660** | **79.88%** | **2.6s** | [`submission_ensemble.csv`](submission_ensemble.csv) |

---

## ⚡ Mobile Compute & Efficiency Benchmark

All models were executed directly inside **Android Termux** on an ARM64 CPU without cloud GPUs:

* **Ultra-Fast Linear Baselines:** TF-IDF + Logistic / Ridge trains across all 5 folds in **2.2 to 2.7 seconds**, proving that high-performing NLP classification on mobile devices does not require massive 10GB transformer models.
* **Native CatBoost Trees:** CatBoost natively processes raw text features into sub-token dictionaries and gradient-boosted trees in ~2.6 minutes on mobile CPU, reaching `0.7560` F1.
* **Ensemble Blending:** Blending probability distributions across disparate hypothesis classes (Generative Naive Bayes + Discriminative Logistic Regression + Regularized Ridge) pushes 5-fold OOF F1 to **`0.7660`** with balanced precision (`0.7655`) and recall (`0.7664`).

---

## 🔬 Key Engineering Insights

### 1. Preprocessing Keywords as Semantic Anchors
Tweets often lack explicit syntax, but Kaggle's metadata provides `keyword` (e.g., `forest%20fire`, `evacuation`). URL-decoding `%20` and prefixing keywords directly to text acts as a domain-level semantic anchor, boosting TF-IDF n-gram overlap significantly across disaster tweets.

### 2. Threshold Calibration for Asymmetric F1 Optimization
Standard classification thresholds assume a default cutoff of `p >= 0.50`. However, F1-score balances precision and recall harmonically:

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Calibrating the decision threshold to `t = 0.43 - 0.45` shifts marginal positive predictions into the positive class, boosting F1 by over `+0.007` to `+0.010` points on out-of-fold validation without changing model weights.

---

## 🚀 How to Reproduce

### 1. Setup Environment
```bash
git clone https://github.com/malcolmlow/nlp-getting-started.git
cd nlp-getting-started
pip install -r requirements.txt
```

### 2. Run Experiments
```bash
python3 exp01_baselines.py
python3 exp02_countvec_naive_bayes.py
python3 exp03_tfidf_linear_models.py
python3 exp04_catboost_text.py
python3 exp05_ensemble.py
```

All generated submission files are saved to root directory and validated against Kaggle test IDs.
