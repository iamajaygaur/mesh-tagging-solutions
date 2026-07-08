# MeSH Tagging — sp26-dl

**Competition:** [MESH-tagging-task-sp26-dl on Kaggle](https://www.kaggle.com/competitions/mesh-tagging-task-sp26-dl)  
**Task:** Multi-label classification — assign one or more MeSH (Medical Subject Heading) terms to PubMed article abstracts.  
**Metric:** micro-F1 score  

---

## Solution Overview

This solution fine-tunes a biomedical pre-trained BERT model (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) on 100,000 annotated PubMed articles to predict MeSH Major terms for unseen articles.

### Approach

1. **Text input:** concatenate article `title` + `abstractText` into a single string
2. **Tokenization:** HuggingFace AutoTokenizer, max_length=256 (512 recommended with more VRAM)
3. **Model:** PubMedBERT with a linear classification head over the `[CLS]` token
4. **Loss:** `BCEWithLogitsLoss` (binary cross-entropy per label)
5. **Threshold:** sigmoid output thresholded at 0.4 (tuned on test set)
6. **Evaluation:** micro-F1 via `sklearn.metrics.f1_score(average='micro')`

### Baseline (TF-IDF + LinearSVC)
- TF-IDF vectorizer (100k features, bigrams) + OneVsRest LinearSVC
- Fast to run, no GPU needed
- Expected micro-F1: ~0.55–0.65

### Main Model (PubMedBERT)
- Fine-tuned for 3 epochs, AdamW optimizer, lr=2e-5, linear warmup
- Expected micro-F1: ~0.70–0.80

---

## Repository Structure

```
mesh_tagging_solution.ipynb   # Main notebook (all cells)
requirements.txt              # Python dependencies
README.md                     # This file
```

> **Note:** Data files are NOT included. Download them directly from the Kaggle competition page.

---

## How to Reproduce

### On Kaggle (recommended)
1. Go to the competition notebook page and create a new notebook
2. Enable GPU accelerator (P100 or T4) in Settings
3. Upload `mesh_tagging_solution.ipynb`
4. The dataset files are auto-mounted at `/kaggle/input/mesh-tagging-task-sp26-dl/`
5. Run all cells — `submission_bert.csv` will appear in the output

### Locally
```bash
# 1. Clone / download this repo
# 2. Install dependencies
pip install -r requirements.txt

# 3. Place data files in the same folder or update paths in Cell 3:
#    TRAIN_PATH = 'path/to/training-set-100000.csv'
#    TEST_PATH  = 'path/to/test-set-20000.csv'
#    JUDGE_PATH = 'path/to/judge-set-10001-unannotated.csv'

# 4. Launch notebook
jupyter notebook mesh_tagging_solution.ipynb
```

---

## Data

| File | Rows | Description |
|------|------|-------------|
| `training-set-100000.csv` | 100,000 | Annotated articles (pmid, title, abstractText, meshMajor) |
| `test-set-20000.csv` | 20,000 | Annotated test articles for local evaluation |
| `judge-set-10001-unannotated.csv` | 10,001 | Unannotated — predict these for submission |
| `sample-submission.csv` | 10,001 | Example submission format |

Source: PubMed / MEDLINE via [https://pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov)

---

## Submission Format

```
pmid,meshMajor
16854706,"['Risk Assessment', 'Sensitivity and Specificity']"
12943287,"['Anti-Bacterial Agents']"
...
```

- `meshMajor` values are **case-sensitive**
- Labels do **not** need to be ordered
- Submit `submission_bert.csv` (BERT model output)

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext |
| MAX_LEN | 256 (512 recommended) |
| BATCH_SIZE | 16 |
| EPOCHS | 3 |
| Learning rate | 2e-5 |
| Threshold | 0.4 (tuned) |
| Optimizer | AdamW, weight_decay=0.01 |
| Scheduler | Linear warmup (10% of steps) |

---

## Ways to Improve Score

| Technique | Expected gain |
|-----------|--------------|
| Increase MAX_LEN to 512 | +1–2% |
| Per-label threshold tuning | +1–3% |
| Switch to BioLinkBERT-large | +2–5% |
| Train 5+ epochs with early stopping | +1–3% |
| Class-weighted BCE loss | +1–2% |
| Ensemble BERT + TF-IDF | +1–3% |

---

## External Resources

- Pre-trained model: [microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) (HuggingFace Hub — downloaded at runtime, not included)
- No other external datasets or pre-trained weights used

---

## Requirements

- Python 3.10+
- CUDA GPU recommended (Kaggle T4/P100 sufficient)
- ~6 GB VRAM minimum for BATCH_SIZE=16, MAX_LEN=256

---

*Individual assignment — sp26-dl, Spring 2026*
