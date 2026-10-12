# Historical Evaluation Images Backup

This directory archives the historical evaluation figures and diagrams originally produced in Google Colab by Pratik Sharma.

---

## Maintainer Information

* **Maintainer:** Ayush Raj (<ayushrajcodes0407@gmail.com>)
* **Repository:** [ayushrajcodes0407/Sarcasm_Analysis_NLP_Project](https://github.com/ayushrajcodes0407/Sarcasm_Analysis_NLP_Project)

---

## Catalog of Historical Images

| Filename | Type | Historical Context | Status in Project |
| :--- | :--- | :--- | :--- |
| `model.png` | Architecture Diagram | Keras BiLSTM network plot (Embedding -> BiLSTM -> MaxPool -> Dense) | **Preserved as Canonical Architecture in `images/model.png`** |
| `sequential_model.JPG` | Architecture Summary | Screenshot of Keras Sequential layer summary | **Preserved as Historical Asset** |
| `confusion_matrix.JPG` | Evaluation Plot | Test evaluation heatmap (Accuracy: ~86%) from original Colab run | **Preserved as Historical Baseline** |
| `monitoring_performance.JPG` | Evaluation Plot | Training history curves (Loss & Accuracy across 6 epochs) | **Preserved as Historical Baseline** |
| `evaluate_randomly.JPG` | Evaluation Plot | Console output of 5 sample headline predictions | **Preserved as Historical Baseline** |
| `word_cloud_sarcastic.png` | Exploratory Visual | Word cloud generated from sarcastic headlines in original dataset | **Preserved as Historical Baseline** |
| `word_cloud_non_sarcastic.png` | Exploratory Visual | Word cloud generated from non-sarcastic headlines in original dataset | **Preserved as Historical Baseline** |

---

## Usage Notice

In accordance with scientific integrity guidelines:
* These historical figures are **preserved intact** and not fabricated or doctored.
* When the raw dataset (`data/raw/Sarcasm_Headlines_Dataset.json`) is placed in `data/raw/`, fresh evaluation curves will be generated using `python src/generate_evaluation_figures.py`.
