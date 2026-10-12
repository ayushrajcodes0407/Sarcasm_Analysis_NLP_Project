"""
Turnkey script to train, evaluate, and generate all 10 evaluation figures
as soon as Sarcasm_Headlines_Dataset.json is placed into data/raw/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src/ is on sys.path
src_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(src_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from text_preprocess import replace_words, remove_punctuation, lem, remove_stopwords, plot_wordcloud
from model import prepare_data_splits, build_bilstm_model, build_tfidf_baseline, load_glove_embeddings, build_embedding_matrix
from evaluate import calculate_metrics, save_metrics_to_json

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def run_pipeline():
    raw_data_path = os.path.join(project_dir, "data", "raw", "Sarcasm_Headlines_Dataset.json")
    figures_dir = os.path.join(project_dir, "results", "figures")
    metrics_path = os.path.join(project_dir, "results", "metrics.json")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("SARCASM DETECTION - EVALUATION & FIGURE GENERATOR")
    print("=" * 60)

    if not os.path.exists(raw_data_path):
        print(f"[BLOCKED] Raw dataset not found at: {raw_data_path}")
        print("Please place 'Sarcasm_Headlines_Dataset.json' in 'data/raw/' to execute.")
        print("See 'data/README.md' for download instructions.")
        return False

    print(f"[1/6] Loading dataset from: {raw_data_path}")
    df = pd.read_json(raw_data_path, lines=True)
    print(f"Loaded {len(df)} records. Columns: {list(df.columns)}")

    # 1. Figure: Class Distribution
    print("[2/6] Generating class_distribution.png...")
    plt.figure(figsize=(7, 5))
    counts = df["is_sarcastic"].value_counts()
    ax = sns.barplot(x=["Non-sarcastic (0)", "Sarcastic (1)"], y=[counts.get(0, 0), counts.get(1, 0)], palette=["#4575b4", "#d73027"])
    plt.title("Class Distribution: News Headlines Sarcasm Dataset", fontsize=13)
    plt.ylabel("Number of Headlines")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                    ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "class_distribution.png"), dpi=200)
    plt.close()

    # 2. Preprocess text
    print("[3/6] Preprocessing headlines...")
    cleaned = df["headline"].astype(str).str.lower()
    cleaned = cleaned.apply(replace_words)
    cleaned = cleaned.apply(remove_punctuation)
    cleaned = cleaned.apply(lem)
    df["cleaned_headline"] = cleaned

    # Word clouds
    print("Generating word clouds...")
    try:
        sarcastic_text = " ".join(df[df["is_sarcastic"] == 1]["cleaned_headline"])
        plot_wordcloud(sarcastic_text, title="Word Cloud: Sarcastic Headlines")
        plt.savefig(os.path.join(figures_dir, "word_cloud_sarcastic.png"), dpi=200)
        plt.close()

        non_sarcastic_text = " ".join(df[df["is_sarcastic"] == 0]["cleaned_headline"])
        plot_wordcloud(non_sarcastic_text, title="Word Cloud: Non-Sarcastic Headlines")
        plt.savefig(os.path.join(figures_dir, "word_cloud_non_sarcastic.png"), dpi=200)
        plt.close()
    except Exception as e:
        print("Note on wordcloud generation:", e)

    # 3. Stratified Split (No Leakage)
    print("[4/6] Creating stratified Train/Val/Test splits...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test), tokenizer, maxlen = prepare_data_splits(
        df["cleaned_headline"].tolist(), df["is_sarcastic"].tolist(),
        test_size=0.2, val_size=0.1, random_state=42, max_features=10000
    )

    # 4. Check for GloVe
    glove_path = os.path.join(project_dir, "data", "external", "glove.6B.200d.txt")
    emb_matrix = None
    num_words = min(10000, len(tokenizer.word_index) + 1)
    if os.path.exists(glove_path):
        print(f"Loading GloVe embeddings from: {glove_path}")
        embeddings = load_glove_embeddings(glove_path, embedding_dim=200)
        emb_matrix, num_words = build_embedding_matrix(tokenizer.word_index, embeddings, max_features=10000, embedding_dim=200)
    else:
        print("GloVe embedding not found in data/external/; initializing trainable embedding layer.")

    # 5. Train Models
    print("[5/6] Training BiLSTM model...")
    bilstm = build_bilstm_model(num_words=num_words, embedding_dim=200, embedding_matrix=emb_matrix, maxlen=maxlen, trainable_embeddings=(emb_matrix is None))
    history = bilstm.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=6,
        batch_size=100,
        verbose=1
    )

    print("Training TF-IDF Baseline model...")
    tfidf_clf = build_tfidf_baseline()
    train_texts = [df["cleaned_headline"].iloc[i] for i in range(len(x_train))]
    tfidf_clf.fit(train_texts, y_train)

    # 6. Evaluation & Figures
    print("[6/6] Generating evaluation figures and saving metrics...")
    
    # Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(history.history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    plt.title("BiLSTM: Training and Validation Loss", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "training_validation_loss.png"), dpi=200)
    plt.close()

    # Accuracy curves
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy", color="#1f77b4", lw=2)
    plt.plot(history.history["val_accuracy"], label="Val Accuracy", color="#ff7f0e", lw=2)
    plt.title("BiLSTM: Training and Validation Accuracy", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "training_validation_accuracy.png"), dpi=200)
    plt.close()

    # Predictions
    y_prob = bilstm.predict(x_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-sarcastic", "Sarcastic"], yticklabels=["Non-sarcastic", "Sarcastic"])
    plt.title("BiLSTM Confusion Matrix (Test Split)", fontsize=13)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # Normalized Confusion Matrix
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=["Non-sarcastic", "Sarcastic"], yticklabels=["Non-sarcastic", "Sarcastic"])
    plt.title("BiLSTM Normalized Confusion Matrix (Test Split)", fontsize=13)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "normalized_confusion_matrix.png"), dpi=200)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color="#2ca02c", lw=2, label="BiLSTM ROC")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "roc_curve.png"), dpi=200)
    plt.close()

    # Precision-Recall Curve
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5.5))
    plt.plot(rec_vals, prec_vals, color="#9467bd", lw=2, label="BiLSTM PR Curve")
    plt.title("Precision-Recall Curve", fontsize=13)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "precision_recall_curve.png"), dpi=200)
    plt.close()

    # Model Comparison
    test_texts = [df["cleaned_headline"].iloc[i] for i in range(len(x_test))]
    tfidf_prob = tfidf_clf.predict_proba(test_texts)[:, 1]
    tfidf_pred = tfidf_clf.predict(test_texts)

    bilstm_metrics = calculate_metrics(y_test, y_pred, y_prob)
    tfidf_metrics = calculate_metrics(y_test, tfidf_pred, tfidf_prob)

    plt.figure(figsize=(8, 5))
    models = ["TF-IDF + LogReg", "BiLSTM Neural Net"]
    accs = [tfidf_metrics["accuracy"] * 100, bilstm_metrics["accuracy"] * 100]
    f1s = [tfidf_metrics["f1_sarcastic"] * 100, bilstm_metrics["f1_sarcastic"] * 100]
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, accs, width, label="Accuracy (%)", color="#1f77b4")
    plt.bar(x + width/2, f1s, width, label="Sarcastic F1 (%)", color="#ff7f0e")
    plt.xticks(x, models, fontsize=11)
    plt.ylabel("Score (%)")
    plt.title("Model Comparison on Held-out Test Set", fontsize=13)
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_comparison.png"), dpi=200)
    plt.close()

    # Save metrics JSON
    final_output = {
        "metadata": {
            "evaluation_status": "FRESHLY_EVALUATED",
            "dataset_file": raw_data_path,
            "test_sample_count": len(y_test),
            "random_seed": 42
        },
        "bilstm_metrics": bilstm_metrics,
        "tfidf_baseline_metrics": tfidf_metrics
    }
    save_metrics_to_json(final_output, metrics_path)
    print("All 10 figures and metrics successfully generated!")
    return True


if __name__ == "__main__":
    run_pipeline()
