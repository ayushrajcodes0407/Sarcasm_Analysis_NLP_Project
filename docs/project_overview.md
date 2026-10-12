# Scientific Audit & NLP Pipeline Overview

## 1. Executive Summary

This document provides a comprehensive technical audit, methodology analysis, and scientific reproducibility report for the **Sarcasm Analysis NLP Project**.

* **Maintainer:** **Ayush Raj** (<ayushrajcodes0407@gmail.com>)
* **Repository:** [ayushrajcodes0407/Sarcasm_Analysis_NLP_Project](https://github.com/ayushrajcodes0407/Sarcasm_Analysis_NLP_Project)
* **Code & Pipeline Quality:** **PASS** (Modularized, 12 unit tests passing, data leakage resolved, bugs fixed).
* **Dataset Availability:** **PENDING LOCAL DOWNLOAD** (`Sarcasm_Headlines_Dataset.json` is not bundled in Git; download required).
* **Embedding Availability:** **OPTIONAL** (`glove.6B.200d.txt` is optional; pipeline supports trainable embeddings).

---

## 2. End-to-End Workflow

```
Raw Input Text
      │
      ▼
Text Validation & Cleaning (src/text_preprocess.py)
• Lowercase normalization
• Contraction dictionary expansion
• ASCII punctuation removal
• Negation-preserving stopword filtering ('not' retained)
• WordNet POS-aware lemmatization
      │
      ▼
Stratified 3-Way Partitioning (70% Train / 10% Val / 20% Test)
      │
      ▼
Feature Extraction / Sequence Preparation
┌───────────────────────────────┴───────────────────────────────┐
▼                                                               ▼
TF-IDF Vectorization                                 Keras Tokenizer (Train Only)
• Word & Bigram n-grams                              • Vocabulary size: 10,000
• Max features: 10,000                               • Padding: post, maxlen=40
      │                                                         │
      ▼                                                         ▼
Logistic Regression Classifier                       BiLSTM Neural Network
• Balanced class weights                             • Embedding: GloVe 200d
• Max iterations: 1,000                              • BiLSTM: 128 units
      │                                              • Global Max Pooling (1D)
      │                                              • Dense stack: 128 -> 64 -> 1
      │                                                         │
      └───────────────────────────────┬─────────────────────────┘
                                      ▼
Scientific Evaluation (src/evaluate.py)
• Accuracy, Precision, Recall, F1, Macro F1, Weighted F1
• True vs. Predicted Confusion Matrix
• ROC-AUC and Precision-Recall Curves
• Machine-readable JSON export (results/metrics.json)
```

---

## 3. Forensic Baseline Audit Findings

A forensic audit of the historical notebook and preprocessing scripts identified several critical algorithmic and engineering flaws:

| Component | Historical Flaw | Severity | Status | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **Tokenizer Fitting** | `Tokenizer.fit_on_texts()` was executed across the *entire* dataset before `train_test_split()`. | **High (Data Leakage)** | **RESOLVED** | Modularized in `src/model.py:prepare_data_splits()`. Data is partitioned into Train, Validation, and Test sets *prior* to fitting the tokenizer strictly on training texts. |
| **Metric Inversion** | Legacy script called `classification_report(y_pred, y_test)` instead of `(y_true, y_pred)`. | **High (Inverted Metrics)** | **RESOLVED** | Inverted precision and recall were corrected in `src/evaluate.py:calculate_metrics()` and notebook Cell 47. |
| **Character Lemmatization** | `lemmatize(text)` in `text_preprocess.py` iterated over string characters instead of words (`''.join([lemmatizer.lemmatize(w) for w in text])`). | **Medium (Logic Bug)** | **RESOLVED** | Fixed in `src/text_preprocess.py` to iterate over whitespace-split tokens (`' '.join([lemmatizer.lemmatize(w) for w in str(text).split()])`). |
| **Keras Callback Syntax** | `lr_r = ReduceLROnPlateau(...),` contained a trailing comma, making `lr_r` a tuple and causing `TypeError` in modern Keras. | **Medium (Syntax/Compatibility)** | **RESOLVED** | Removed trailing comma in notebook Cell 40 and modularized in `src/model.py`. |
| **Encoding Error** | `open(EMBEDDING_FILE)` lacked explicit UTF-8 encoding, causing Windows `cp1252` decoding crashes on GloVe vector symbols. | **Medium (Portability)** | **RESOLVED** | Explicit `encoding='utf-8'` added to `src/model.py:load_glove_embeddings()` and notebook Cell 35. |
| **Validation Plot Mislabeled** | Legacy plot plotted `val_loss` and `val_accuracy` but labeled the legend as `['Train', 'Test']`. | **Low (Visual Mislable)** | **RESOLVED** | Relabeled to `['Train', 'Validation']` in Cell 45. |
| **OS Shell Dependency** | Notebook contained `!rm -R log/` which fails on Windows PowerShell/CMD. | **Low (Portability)** | **RESOLVED** | Replaced with cross-platform Python `shutil.rmtree('log', ignore_errors=True)`. |

---

## 4. Historical Baseline Performance Record

The historical Google Colab GPU execution produced the following metrics on the held-out test split (5,342 samples):

* **Sample Distribution:** Total Samples: 26,709 | Test Split: 5,342 (20%)
* **Class Balance:** Class 0 (Non-sarcastic): 2,972 (55.6%) | Class 1 (Sarcastic): 2,370 (44.4%)
* **Historical Metrics:**
  * Test Accuracy: **85.6%**
  * Non-Sarcastic (Class 0): Precision 0.87, Recall 0.88, F1 0.88
  * Sarcastic (Class 1): Precision 0.85, Recall 0.84, F1 0.84
  * Macro Average F1: **0.86**
  * Weighted Average F1: **0.86**

*(Saved in [`results/metrics.json`](../results/metrics.json)).*

---

## 5. Automated Test Suite Execution

The test suite validates data split isolation, contraction expansion, lemmatization, metric calculation, and model compilation:
* **`tests/test_text_preprocess.py`:** 8 unit tests covering preprocessing routines.
* **`tests/test_model_pipeline.py`:** 4 unit tests covering leak-free data splitting, metric math, BiLSTM instantiation, and TF-IDF baseline.
* **Test Result:** **12 passed in ~11 seconds (100% pass rate)**.
