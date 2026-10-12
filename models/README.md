# Model Architecture & Specifications

This directory documents the neural network and baseline architectures implemented in the **Sarcasm Analysis NLP Project**.

---

## Maintainer Information

* **Maintainer:** Ayush Raj (<ayushrajcodes0407@gmail.com>)
* **Repository:** [ayushrajcodes0407/Sarcasm_Analysis_NLP_Project](https://github.com/ayushrajcodes0407/Sarcasm_Analysis_NLP_Project)

---

## 1. Deep Bidirectional LSTM Architecture

Implemented in [`src/model.py:build_bilstm_model()`](../src/model.py):

```
Layer (type)                     Output Shape          Param #     Notes
=================================================================================
Embedding (Embedding)            (None, 40, 200)       2,000,000   Pretrained GloVe (trainable=False/True)
Bidirectional(LSTM)              (None, 40, 256)       336,896     128 units per direction (return_sequences=True)
GlobalMaxPooling1D               (None, 256)           0           Temporal max pooling
Dropout                          (None, 256)           0           Dropout rate = 0.5
Dense (Dense)                    (None, 128)           32,896      ReLU activation
Dropout                          (None, 128)           0           Dropout rate = 0.5
Dense (Dense)                    (None, 64)            8,256       ReLU activation
Dropout                          (None, 64)            0           Dropout rate = 0.5
Dense (Dense)                    (None, 1)             65          Sigmoid binary classification
=================================================================================
Total params: 2,378,113 | Trainable params: 378,113 (with static GloVe)
```

### Architecture Diagram
Visualized in [`images/model.png`](../images/model.png).

### Hyperparameters & Training Configuration
* **Loss Function:** Binary Cross-Entropy (`binary_crossentropy`)
* **Optimizer:** Adam (`learning_rate=0.001`)
* **Batch Size:** 100
* **Epochs:** 6
* **Callbacks:**
  * `EarlyStopping(monitor='val_loss', mode='min', patience=10)`
  * `ModelCheckpoint('sarcasm_detector.keras', monitor='val_loss', save_best_only=True)`
  * `ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)`

---

## 2. Baseline Model: TF-IDF + Logistic Regression

Implemented in [`src/model.py:build_tfidf_baseline()`](../src/model.py):
* **Feature Extraction:** `TfidfVectorizer(max_features=10000, ngram_range=(1, 2))`
* **Classification Algorithm:** `LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')`
* **Purpose:** Provides a deterministic, lightweight benchmark to assess the performance gain of sequence modeling.

---

## 3. Tokenization & Sequence Preparation

* **Vocabulary:** Capped at 10,000 tokens (`max_features=10000`).
* **Out-of-Vocabulary Token:** `<OOV>` handles unseen words during validation and inference.
* **Sequence Length:** Padded or truncated to length `40` (`padding='post'`, `truncating='post'`).
* **Data Leakage Prevention:** The tokenizer is fit **strictly on training texts**.

---

## 4. Inference & Model Usage

To load and run inference programmatically:

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.text_preprocess import replace_words, remove_punctuation, lem

# 1. Clean raw headline
headline = "Local weather forecaster accurately predicts rain five minutes after it starts"
clean_text = lem(remove_punctuation(replace_words(headline.lower())))

# 2. Tokenize and pad (using fitted tokenizer)
seq = tokenizer.texts_to_sequences([clean_text])
padded = pad_sequences(seq, maxlen=40, padding="post")

# 3. Predict probability
prob = float(model.predict(padded, verbose=0)[0][0])
is_sarcastic = int(prob >= 0.5)

print(f"Sarcasm Probability: {prob:.2%}")
print(f"Prediction: {'Sarcastic' if is_sarcastic else 'Non-sarcastic'}")
```

---

## 5. Storage & Versioning Notice

In accordance with open-source repository best practices:
* Large binary checkpoint files (`*.h5`, `*.keras`, `*.pt`) are excluded from Git via `.gitignore`.
* Train a fresh checkpoint locally using `python src/generate_evaluation_figures.py`.
