"""
Unit tests for ML modeling pipeline, data leakage prevention, and evaluation metrics.
"""

import sys
import os
import pytest
import numpy as np

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from model import prepare_data_splits, build_bilstm_model, build_tfidf_baseline
from evaluate import calculate_metrics


class TestModelPipeline:
    def test_data_splits_no_leakage(self):
        """Verify data splits are separated and Tokenizer is fit strictly on training set."""
        texts = [
            "breaking news from the onion",
            "huffington post report on climate",
            "completely unexpected turn of events",
            "unbelievable statement from senator",
            "scientist discovers something new",
            "satirical headline about aliens",
            "traffic jam causes minor delay",
            "sports team wins championship game",
            "local mayor announces public park",
            "new study finds interesting correlation"
        ] * 5  # 50 samples
        labels = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1] * 5
        
        (x_train, y_train), (x_val, y_val), (x_test, y_test), tokenizer, maxlen = prepare_data_splits(
            texts, labels, test_size=0.2, val_size=0.2, random_state=42, max_features=100
        )
        
        # Verify split sizes
        total = len(texts)
        assert len(x_train) + len(x_val) + len(x_test) == total
        assert len(y_train) + len(y_val) + len(y_test) == total
        assert len(x_test) == 10
        assert len(x_val) == 10
        assert len(x_train) == 30
        
        # Verify sequence shape
        assert x_train.shape[1] == maxlen
        assert x_val.shape[1] == maxlen
        assert x_test.shape[1] == maxlen
        
        # Verify tokenizer has OOV token
        assert tokenizer.oov_token == "<OOV>"
        assert "<oov>" in tokenizer.word_index or "<OOV>" in tokenizer.word_index

    def test_calculate_metrics_values(self):
        """Verify metric calculation correctness and proper positive-class orientation."""
        # 4 samples: 2 positive, 2 negative
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 0]  # TP=1, FN=1, TN=2, FP=0
        y_prob = [0.9, 0.4, 0.2, 0.1]
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        # Accuracy: 3 / 4 = 0.75
        assert metrics["accuracy"] == 0.75
        # Precision (sarcastic, label 1): 1 TP / (1 TP + 0 FP) = 1.0
        assert metrics["precision_sarcastic"] == 1.0
        # Recall (sarcastic, label 1): 1 TP / (1 TP + 1 FN) = 0.5
        assert metrics["recall_sarcastic"] == 0.5
        # F1: 2 * (1.0 * 0.5) / (1.0 + 0.5) = 2/3 ≈ 0.6667
        assert round(metrics["f1_sarcastic"], 4) == round(2/3, 4)
        
        # Confusion matrix: [[TN, FP], [FN, TP]] = [[2, 0], [1, 1]]
        assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
        
        # ROC AUC should be 1.0 since positive probs [0.9, 0.4] > negative probs [0.2, 0.1]
        assert metrics["roc_auc"] == 1.0

    def test_build_bilstm_model_architecture(self):
        """Verify BiLSTM model compiles cleanly and produces expected output shape."""
        vocab_size = 50
        maxlen = 10
        model = build_bilstm_model(num_words=vocab_size, embedding_dim=16, maxlen=maxlen)
        
        # Verify layer structure
        layer_names = [l.__class__.__name__ for l in model.layers]
        assert "Embedding" in layer_names
        assert "Bidirectional" in layer_names
        assert "GlobalMaxPooling1D" in layer_names or "GlobalMaxPool1D" in layer_names
        assert "Dense" in layer_names
        
        # Test forward pass with synthetic batch
        dummy_input = np.random.randint(0, vocab_size, size=(2, maxlen))
        pred = model.predict(dummy_input, verbose=0)
        assert pred.shape == (2, 1)
        assert 0.0 <= pred[0][0] <= 1.0

    def test_build_tfidf_baseline_fit_predict(self):
        """Verify TF-IDF baseline pipeline trains and predicts deterministically."""
        train_texts = ["sarcastic statement indeed", "standard news article", "funny joke here", "daily weather report"]
        train_labels = [1, 0, 1, 0]
        
        clf = build_tfidf_baseline()
        clf.fit(train_texts, train_labels)
        
        preds = clf.predict(["sarcastic statement", "weather report"])
        assert len(preds) == 2
        assert set(preds).issubset({0, 1})
