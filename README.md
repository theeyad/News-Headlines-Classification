# News Headline Classification

This repository contains an interactive Jupyter notebook that demonstrates an end-to-end workflow for classifying news headlines using NLP techniques. The primary artifact is the notebook [NLP.ipynb](NLP.ipynb), which includes data loading, preprocessing, model training, evaluation, and inference examples.

## Table of Contents

- **Overview:** Short project description and goals
- **Requirements:** Software, Python version, and packages
- **Data:** Expected dataset format and acquisition
- **Notebook Structure:** What each notebook section does
- **Run Instructions:** How to run the notebook locally
- **Model & Evaluation:** Models used and evaluation metrics
- **Reproducibility:** Seeds, environment capture, and tips
- **Troubleshooting:** Common issues and fixes
- **Citations & License:** Sources and licensing

## Overview

This project demonstrates techniques for classifying news headlines into categories (for example: politics, sports, tech, entertainment). The goal is to provide a clear, reproducible pipeline that can be adapted to new datasets or extended to more advanced models (e.g., transformer-based architectures).

## Requirements

- **Python:** 3.8+ recommended
- **Jupyter:** `jupyterlab` or `notebook`
- **Suggested packages:** scikit-learn, pandas, numpy, matplotlib, seaborn, nltk, spacy, joblib, transformers (optional)

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows cmd
.\.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
```

Install the common dependencies:

```bash
pip install -U pip
pip install jupyterlab pandas numpy scikit-learn matplotlib seaborn nltk spacy joblib
# Optional (if you use transformers in the notebook):
pip install transformers torch
```

Tip: If you want to pin exact package versions, create a `requirements.txt` after testing with:

```bash
pip freeze > requirements.txt
```

## Data

- **Expected format:** a CSV (or similar tabular file) with at least two columns: `headline` and `label` (or `target`).
- **File location:** place your dataset under a new `data/` directory, e.g. `data/news_headlines.csv`.

If you don't have a dataset, the notebook contains a small sample or placeholder cell showing how to load a CSV and how to adapt column names.

## Notebook Structure ([NLP.ipynb](NLP.ipynb))

The notebook is organized into the following logical sections:

- **1. Setup & Imports:** load Python packages, set random seed, and configure plotting.
- **2. Data Loading:** read CSV files, show sample rows, and validate expected columns.
- **3. Exploratory Data Analysis (EDA):** distributions of labels, headline length, and common words.
- **4. Preprocessing:** tokenization, lowercasing, stopword removal, stemming/lemmatization (configurable).
- **5. Feature Engineering:** TF-IDF vectorization (with n-grams), optional word embeddings or transformer features.
- **6. Model Training:** train baseline models (Logistic Regression, Naive Bayes, or optionally an MLP/Transformer).
- **7. Evaluation:** compute accuracy, precision, recall, F1-score, confusion matrix, and class-wise metrics.
- **8. Inference / Prediction:** how to run the model on new headlines and save results.
- **9. Export & Reuse:** save trained model with `joblib` and load it for inference.

Each section includes runnable code cells and short explanatory notes describing choices and trade-offs.

## Running the Notebook Locally

Start JupyterLab or Notebook in the repo root and open `NLP.ipynb`:

```bash
# from repo root
jupyter lab
# or
jupyter notebook
```

Then open the notebook and run cells in order. If you modify cells that change variable names or flow, restart the kernel and run all cells to ensure reproducibility.

### Running non-interactively

If you want to execute the notebook end-to-end non-interactively, you can use `papermill` or `nbconvert`:

```bash
pip install papermill
papermill NLP.ipynb NLP_run.ipynb
```

## Models & Evaluation

- **Baseline approach:** TF-IDF vectorization + linear classifier (Logistic Regression or Multinomial Naive Bayes).
- **Advanced options:** fine-tune a transformer (e.g., `distilbert`) for better accuracy at higher compute cost.
- **Metrics reported:** accuracy, precision, recall, F1-score (macro & micro), and a confusion matrix for qualitative analysis.

Notes on evaluation:

- Use stratified train/test splits for imbalanced classes: `train_test_split(..., stratify=y)`.
- Prefer cross-validation (`StratifiedKFold`) for robust metric estimates when dataset is small.

## Reproducibility

- Set the seed at the top of the notebook (example: `SEED = 42`) and seed numpy, random, and torch if used.
- Record package versions with `pip freeze` and include `requirements.txt` for exact reproduction.
- Save any preprocessing artifacts (e.g., `TfidfVectorizer` object) using `joblib.dump()` so the same transformation is applied at inference.

## Expected Outputs & Interpretation

- The notebook will show a dataset sample and label distribution plots.
- The model training cells will print cross-validation or hold-out scores and a final test set classification report.
- Use the confusion matrix to inspect which classes are commonly confused and refine preprocessing or class weighting accordingly.
