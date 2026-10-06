# Kaggle NLP with Disaster Tweets — Pocket Data Science V

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://python.org)
[![Platform: Android Termux](https://img.shields.io/badge/Platform-Android%20Termux-black?logo=android&logoColor=white)](https://termux.dev)
[![Agent: Google Antigravity CLI](https://img.shields.io/badge/AI%20Agent-Google%20Antigravity%20CLI-4285F4?logo=google&logoColor=white)](https://antigravity.google)
[![Kaggle Challenge](https://img.shields.io/badge/Kaggle-Natural%20Language%20Processing%20with%20Disaster%20Tweets-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/nlp-getting-started)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, systematic natural language processing study for the [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) challenge. Built, trained, cross-validated, and ensembled entirely on a consumer **Android smartphone CPU** using **Termux PRoot** and **Google Antigravity CLI (`agy`)**.

---

## 🏆 The Empirical Experiment Progression

Each experiment establishes a disciplined step up in modeling complexity, evaluated with **Stratified 5-Fold Cross-Validation** and scored on Kaggle's live public leaderboard:

| Exp | Script | Pipeline / Strategy | 5-Fold OOF Acc | Kaggle Public Score | Leaderboard Rank | Rank Delta | Test Submission |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **0a** | [`exp01_baselines.py`](exp01_baselines.py) | Constant Majority Class (All 0s) | 57.03% | `0.57033` | #431 | Baseline | [`submission_majority.csv`](submission_majority.csv) |
| **0b** | [`exp01_baselines.py`](exp01_baselines.py) | Constant Minority Class (All 1s) | 42.97% | `0.42966` | #434 | -3 | [`submission_constant_1.csv`](submission_constant_1.csv) |
| **0c** | [`exp01_baselines.py`](exp01_baselines.py) | Empirical Random Prior (`p=0.430`) | 49.98% | `0.51087` | Floor | — | [`submission_random.csv`](submission_random.csv) |
| **1** | [`exp02_countvec_naive_bayes.py`](exp02_countvec_naive_bayes.py) | CountVectorizer (1,2-gram) + Multinomial NB | 80.13% | `0.79497` | #314 | **+117** | [`submission_countvec_nb.csv`](submission_countvec_nb.csv) |
| **2** | [`exp03_tfidf_linear_models.py`](exp03_tfidf_linear_models.py) | Sublinear TF-IDF + Logistic Regression (`C=1.5`) | 80.82% | `0.79957` | #269 | **+45** | [`submission_tfidf_logistic.csv`](submission_tfidf_logistic.csv) |
| **3** | [`exp04_catboost_text.py`](exp04_catboost_text.py) | CatBoost Native Text Gradient Boosted Trees | 79.81% | `0.77811` | — | — | [`submission_catboost.csv`](submission_catboost.csv) |
| **4** | [`exp05_ensemble.py`](exp05_ensemble.py) | Multi-Model Blend (Logistic + NB + Ridge) | 81.11% | `0.79803` | — | — | [`submission_ensemble.csv`](submission_ensemble.csv) |
| **5** | [`exp06_word_char_tfidf.py`](exp06_word_char_tfidf.py) | Word (1,2) + Char (3,5) FeatureUnion + Logistic | 81.51% | `0.80478` | #221 | **+48** | [`submission_word_char_tfidf.csv`](submission_word_char_tfidf.csv) |
| **6** | [`exp07_dense_transformer_embeddings.py`](exp07_dense_transformer_embeddings.py) | Dense MiniLM Transformer + Sparse TF-IDF Hybrid | 83.06% | `0.82470` | #165 | **+56** | [`submission_top100_candidate.csv`](submission_top100_candidate.csv) |
| **8** | [`exp08_production_ensemble.py`](exp08_production_ensemble.py) | 100% Pure ML Single-Backbone (Zero Overrides/Leakage) | 82.95% | `0.82408` | #166 | — | [`submission_pure_ml_hybrid.csv`](submission_pure_ml_hybrid.csv) |
| **9** | [`exp10_final_submission.py`](exp10_final_submission.py) | Dual-Backbone 768-D Dense (MiniLM + BGE-Small) + Sparse | 83.42% | `0.82899` | #147 | **+19** | [`submission_dual_backbone_pure_ml.csv`](submission_dual_backbone_pure_ml.csv) |
| **10** | [`exp10_final_submission.py`](exp10_final_submission.py) | **5-Model Multi-Loss Hybrid (Dual Dense + Ridge + LR + Joint)** | **83.71%** | **`0.83021`** | **#138** | **+9** | [`submission_dual_ridge_lr_hybrid.csv`](submission_dual_ridge_lr_hybrid.csv) |

*(Total Leaderboard Gain: **+293 places**, climbing from Rank #431 into the **Top 31.5% globally at Rank #138** with 100% Pure Machine Learning, just 0.00429 points away from the Top 100 cutoff `0.8345`).*

---

## ⚡ Mobile Compute & Efficiency Benchmark

All models were trained directly inside **Android Termux** on an ARM64 CPU without cloud GPUs:

* **Word + Character FeatureUnion:** Generating 38,000+ sparse features combining word n-grams `(1, 2)` and subword character n-grams `(3, 5)` trains across 5 folds in **13.7 seconds**, achieving our highest Kaggle score (**`0.80478`**).
* **Ultra-Fast Linear Baselines:** TF-IDF + Logistic Regression trains across 5 folds in **2.7 seconds**, proving that high-performing NLP classification on mobile devices does not require massive 10GB transformer models.
* **Native CatBoost Trees vs Linear Models:** CatBoost natively processes text tokens into decision trees in 221 seconds (`0.77811`), while regularized linear models run 82× faster and score +2.6% higher due to better suited continuous hyperplanes over sparse vocabularies.

---

## 🔬 Key Engineering Insights

### 1. The Live Scoring Metric Truth: Categorical Accuracy
Despite competition overview text describing binary positive-class F1-score, probing the live public leaderboard with Constant 0 (`0.57033`) and Constant 1 (`0.42966`) sums to exactly `0.99999 ≈ 1.00000`. This 1.0 complement rule mathematically proves Kaggle's live public scoring engine evaluates `accuracy_score`, not binary F1.

### 2. Reverse-Engineering the Unseen Test Set
From our baseline probes, the 3,263 unseen test set tweets contain:
* **Safe Tweets (Class 0):** `3,263 × 0.570334 = 1,861` tweets (57.03%)
* **Disaster Tweets (Class 1):** `3,263 × 0.429666 = 1,402` tweets (42.97%)
This matches the training set split (`57.03%` vs `42.97%`) to the third decimal place, proving zero label shift between train and test.

### 3. Subword Character N-Grams for Twitter Noise
Twitter text is riddled with typos, informal slang, hashtags, and concatenated tokens. Adding character n-grams (`analyzer='char_wb'`, `ngram_range=(3, 5)`) via `FeatureUnion` enables the model to match subword roots and morphological stems across out-of-vocabulary misspellings, breaking the 80% mark on the live leaderboard.

---

## 🚀 How to Reproduce

### 1. Setup Environment
```bash
git clone https://github.com/myhlow/nlp-getting-started.git
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
python3 exp06_word_char_tfidf.py
```

All generated submission files are saved to the workspace and validated against Kaggle test IDs.
