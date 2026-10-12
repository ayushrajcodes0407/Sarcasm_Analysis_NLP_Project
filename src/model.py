"""
Model architectures, data splitting, and embedding utilities.
Designed to prevent data leakage and ensure reproducible NLP pipelines.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def prepare_data_splits(texts, labels, test_size=0.2, val_size=0.1, random_state=42, max_features=10000):
    """
    Split data into Train, Validation, and Test sets BEFORE fitting the tokenizer
    to eliminate data leakage between splits.
    
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test), tokenizer, maxlen
    """
    # 1. Stratified split into train_val and test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels, shuffle=True
    )
    
    # 2. Stratified split into train and validation
    rel_val_size = val_size / (1.0 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=rel_val_size, random_state=random_state, stratify=train_val_labels, shuffle=True
    )
    
    # 3. Fit Tokenizer ONLY on training texts (CRITICAL for data leakage prevention)
    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    
    # Determine maxlen from training set lengths (or robust 99th percentile)
    train_lengths = [len(t.split()) for t in train_texts]
    maxlen = max(train_lengths) if train_lengths else 40
    
    # 4. Transform all splits using training-fitted tokenizer
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=maxlen, padding="post", truncating="post")
    x_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=maxlen, padding="post", truncating="post")
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=maxlen, padding="post", truncating="post")
    
    y_train = np.asarray(train_labels, dtype=np.int32)
    y_val = np.asarray(val_labels, dtype=np.int32)
    y_test = np.asarray(test_labels, dtype=np.int32)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), tokenizer, maxlen


def load_glove_embeddings(filepath, embedding_dim=200):
    """
    Safely load pretrained GloVe embeddings from file using UTF-8 encoding.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GloVe embedding file not found at: {filepath}")
    
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            word = parts[0]
            vector = np.asarray(parts[1:], dtype=np.float32)
            if len(vector) == embedding_dim:
                embeddings[word] = vector
    return embeddings


def build_embedding_matrix(word_index, embeddings, max_features=10000, embedding_dim=200):
    """
    Construct embedding matrix mapping tokenizer word indices to GloVe vectors.
    """
    num_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim), dtype=np.float32)
    for word, i in word_index.items():
        if i >= num_words:
            continue
        vector = embeddings.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
    return embedding_matrix, num_words


def build_bilstm_model(num_words, embedding_dim=200, embedding_matrix=None, maxlen=40, trainable_embeddings=False):
    """
    Construct the BiLSTM classification neural network.
    """
    model = Sequential()
    if embedding_matrix is not None:
        model.add(Embedding(
            num_words,
            embedding_dim,
            embeddings_initializer=Constant(embedding_matrix),
            trainable=trainable_embeddings
        ))
    else:
        model.add(Embedding(
            num_words,
            embedding_dim,
            ))
        
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model


def build_tfidf_baseline():
    """
    Lightweight, deterministic TF-IDF + Logistic Regression benchmark model.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))
    ])
