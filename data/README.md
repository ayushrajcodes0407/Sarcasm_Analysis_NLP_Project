# Data Management & Specifications

This directory defines the data architecture, schema, acquisition guidelines, and storage conventions for the **Sarcasm Analysis NLP Project**.

---

## Maintainer Information

* **Maintainer:** Ayush Raj (<ayushrajcodes0407@gmail.com>)
* **Repository:** [ayushrajcodes0407/Sarcasm_Analysis_NLP_Project](https://github.com/ayushrajcodes0407/Sarcasm_Analysis_NLP_Project)

---

## Directory Organization

```
data/
├── raw/         # Original, unmodified raw datasets (e.g., Sarcasm_Headlines_Dataset.json)
├── processed/   # Cleaned, tokenized, or split datasets (train/val/test splits)
├── external/    # Third-party assets (e.g., pretrained GloVe embeddings)
└── README.md    # Data documentation (this file)
```

---

## 1. Primary Dataset: News Headlines for Sarcasm Detection

* **Expected Filename:** `Sarcasm_Headlines_Dataset.json`
* **Expected Location:** `data/raw/Sarcasm_Headlines_Dataset.json`
* **Current Status:** **Not bundled in Git repository** (excluded via `.gitignore` to prevent repository bloat).
* **Provenance:** Compiled by Rishabh Misra and Prahal Arora (2019).
* **Paper Reference:** Misra, Rishabh, and Prahal Arora. *"Sarcasm Detection using Hybrid Neural Network."* arXiv preprint [arXiv:1908.07414](https://arxiv.org/abs/1908.07414).

### Dataset Sources
* **Sarcastic Headlines:** Sourced from satirical publication [*The Onion*](https://theonion.com).
* **Non-Sarcastic Headlines:** Sourced from [*HuffPost*](https://huffpost.com).

### Verified Schema & Attributes

| Field | Data Type | Description | Handling in Pipeline |
| :--- | :--- | :--- | :--- |
| `is_sarcastic` | `int` | Binary classification target: `1` = Sarcastic, `0` = Non-sarcastic | Used as prediction label `y`. |
| `headline` | `string` | The news article headline text | Normalized and tokenized as model feature `X`. |
| `article_link` | `string` | Canonical URL to original publication | Dropped during preprocessing. |

### Data Quality & Preprocessing Handling
* **Missing Values:** Records with missing headlines or labels are dropped during pipeline validation.
* **Duplicate Records:** Duplicate headlines are filtered to prevent data contamination across splits.
* **Class Balance:** Approximately 55.6% non-sarcastic vs. 44.4% sarcastic in version 1 (26,709 total samples).

---

## 2. External Embeddings: Stanford GloVe Vectors

* **Specification:** GloVe 6B word representation, 200 dimensions.
* **Expected Filename:** `glove.6B.200d.txt` (or archive `glove.6B.zip`)
* **Expected Location:** `data/external/glove.6B.200d.txt`
* **Reference:** Pennington, Jeffrey, Richard Socher, and Christopher D. Manning. *"GloVe: Global Vectors for Word Representation."* EMNLP 2014.
* **Optional / Fallback Mode:**
  * When `glove.6B.200d.txt` is present: The BiLSTM model initializes its embedding weights using GloVe and keeps them static (`trainable=False`).
  * When `glove.6B.200d.txt` is absent: The pipeline automatically initializes a trainable embedding layer from scratch (`trainable=True`), allowing model training to proceed without GloVe.

---

## 3. How to Obtain and Set Up Data

Follow these steps to set up the dataset locally:

1. **Obtain the Dataset:**
   * Download `Sarcasm_Headlines_Dataset.json` from the [Kaggle Sarcasm Detection Dataset](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection) or the author's [research repository](https://rishabhmisra.github.io/).
2. **Place the file in `data/raw/`:**
   ```powershell
   # Target destination:
   Sarcasm_Analysis_NLP_Project\dataaw\Sarcasm_Headlines_Dataset.json
   ```
3. **(Optional) Obtain GloVe Embeddings:**
   * Download `glove.6B.zip` from [Stanford NLP GloVe](https://nlp.stanford.edu/projects/glove/).
   * Extract `glove.6B.200d.txt` into:
   ```powershell
   Sarcasm_Analysis_NLP_Project\data\external\glove.6B.200d.txt
   ```

---

## 4. Privacy, Licensing & Versioning Policy

* **Privacy:** The dataset contains public journalistic headlines and contains no private or personally identifiable information (PII).
* **Licensing:** Sourced from public research releases and subject to the respective authors' academic distribution terms.
* **Version Control:** All files matching `data/raw/*`, `data/processed/*`, and `data/external/*` are ignored by Git via `.gitignore` (except `.gitkeep` and `README.md`) to maintain a clean repository.
