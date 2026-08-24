# BC Detect — A Dual-Pathway Breast Cancer Detection Tool

A decision-support tool that flags a breast cancer case as benign or malignant,
intended as a second reader alongside a clinician rather than a replacement for
one. A clinician always makes the final decision.

The system has two pathways. The **classical pathway** trains five interpretable
classifiers on the tabular Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The
**imaging pathway** trains a convolutional neural network on CBIS-DDSM mammography
images, developed as a staged ablation (baseline, regularised, and transfer
learning). Both pathways report through the same clinical metrics and are compared
against published benchmarks. A Streamlit application presents both pathways to a
user with a clear recommendation, a confidence level, and an explanation.

This project was developed as a final-year BSc Computer Science project
(University of London), following Project Idea 2, Deep Learning Breast Cancer
Detection.

## Repository contents

| File | Purpose |
|---|---|
| `classical.py` | Trains and evaluates the five classical WDBC classifiers |
| `prepare_data.py` | Sorts CBIS-DDSM images into class folders by biopsy-confirmed pathology |
| `prototype.py` | Trains and evaluates the imaging CNN (baseline, regularised, transfer, fine-tuned) |
| `predict.py` | Single-image inference from a saved imaging model |
| `metrics.py` | Clinical metrics (accuracy, sensitivity, specificity) from a confusion matrix |
| `app.py` | Streamlit application (pathway chooser, result, benchmarking) |
| `tests/` | Automated pytest suite |

## Requirements

Python 3.9 or later. Install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No GPU is required; the models are small and run on CPU. On Apple Silicon, do not
install `tensorflow-metal`, as it can cause instability with this project; the
pinned CPU build in `requirements.txt` is the supported configuration.

## Running the tests

```bash
python -m pytest tests/ -v
```

## The classical pathway (WDBC)

The Wisconsin dataset ships with scikit-learn, so no download is needed. Train and
evaluate all five classifiers:

```bash
python classical.py
```

This prints each model's cross-validation and held-out test results (accuracy,
sensitivity, specificity, AUC) and saves the trained models to `outputs/`.

## The imaging pathway (CBIS-DDSM)

The imaging pathway uses the JPEG version of CBIS-DDSM, available from Kaggle
(around 6 GB). Because the folder names in the download are scan identifiers rather
than diagnoses, images must be sorted by the pathology labels in the accompanying
CSV files rather than by folder. `prepare_data.py` does this:

```bash
python prepare_data.py --root /path/to/CBIS-DDSM --out data --per_class 200
```

This produces `data/benign/` and `data/malignant/`. Then train the network. The
tier is selected by flag:

```bash
python prototype.py --data_dir data --epochs 15                 # baseline
python prototype.py --data_dir data --epochs 15 --regularised   # regularised tier
python prototype.py --data_dir data --epochs 15 --transfer      # transfer learning (EfficientNet)
python prototype.py --data_dir data --epochs 15 --transfer --finetune   # fine-tuned variant
```

With no `--data_dir`, the script runs on generated synthetic data so the pipeline
can be demonstrated without the dataset. Each run prints an evaluation and saves
the model and plots to `outputs/`.

Classify a single image with a saved model:

```bash
python predict.py --image path/to/mammogram.jpg
```

## The application

With a trained classical model (`python classical.py`) and, for the imaging
pathway, a trained CNN in `outputs/`, launch the app:

```bash
streamlit run app.py
```

It opens in the browser and offers both pathways, a result view with confidence and
an explanation, and a benchmarking view comparing the models.

## Results summary

The classical models reach around 0.95 to 0.98 sensitivity on the WDBC features,
near the published benchmarks. The imaging models are more modest on raw
mammograms: the regularised network is the strongest at 0.799 sensitivity and 0.703
AUC, and transfer learning from a natural-image network did not improve on it. This
contrast between a strong tabular pathway and a harder imaging task, quantified on
comparable footing, is the project's central finding.

## Data attribution

Wisconsin Diagnostic Breast Cancer dataset: Street, Wolberg and Mangasarian (1993),
via scikit-learn. CBIS-DDSM: Lee et al. (2017), a public de-identified research
dataset.
