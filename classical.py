"""
classical.py

The classical pathway of the dual-pathway system. Trains five interpretable
classifiers on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, evaluates
each with stratified k-fold cross-validation and on a held-out test set, and
saves the trained models.

This mirrors the imaging pathway deliberately: the same fixed seed, the same
clinical metrics from metrics.py, and the same convention that class 1 is
malignant, so a false negative is a missed cancer. The point of the pathway is a
fair, reproducible comparison of standard models on tabular cytology features,
reported through the metric that matters clinically (sensitivity) rather than raw
accuracy.

The WDBC data ships inside scikit-learn (load_breast_cancer), so no download is
needed and anyone running this gets the identical 569 samples and 30 features
described by Street et al. (1993).

Usage
    python classical.py                 # train, evaluate, save models
    python classical.py --folds 10      # 10-fold cross-validation
"""

import argparse
import json
import os
import random

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import joblib

from metrics import compute_metrics

SEED = 42
OUT = "outputs"

random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    """Load WDBC from scikit-learn.

    sklearn labels benign as 1 and malignant as 0. This project's convention is
    the opposite, malignant as the positive class 1, so the target is flipped to
    keep sensitivity meaning 'malignant cases caught' consistently across both
    pathways.
    """
    ds = load_breast_cancer()
    X = ds.data
    y = 1 - ds.target  # flip: 1 = malignant (positive), 0 = benign
    return X, y, list(ds.feature_names)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_models():
    """Return the five classifiers, each wrapped in a scaling pipeline.

    Scaling matters for SVM, logistic regression and k-NN, which are
    distance- or gradient-based; the tree ensembles are scale-invariant but the
    uniform pipeline keeps the comparison clean and the preprocessing identical
    across models, which is the fairness the design calls for. Where a model
    exposes class weighting, malignant cases are up-weighted to reflect the
    imbalance and the higher cost of a miss.
    """
    return {
        "SVM": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300,
                                           class_weight="balanced",
                                           random_state=SEED)),
        ]),
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000,
                                       class_weight="balanced",
                                       random_state=SEED)),
        ]),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scale", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=SEED)),
        ]),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def cross_validate(model, X, y, folds):
    """Stratified k-fold cross-validation, returning mean accuracy and AUC.

    Stratification keeps the malignant/benign ratio consistent in every fold, so
    no fold is left thin on the minority class.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    accs, aucs = [], []
    for train_idx, val_idx in skf.split(X, y):
        m = _clone_and_fit(model, X[train_idx], y[train_idx])
        prob = m.predict_proba(X[val_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        cm = confusion_matrix(y[val_idx], pred, labels=[0, 1])
        accs.append(compute_metrics(cm)["accuracy"])
        aucs.append(roc_auc_score(y[val_idx], prob))
    return float(np.mean(accs)), float(np.mean(aucs))


def _clone_and_fit(model, X, y):
    from sklearn.base import clone
    m = clone(model)
    m.fit(X, y)
    return m


def evaluate_on_test(model, X_train, y_train, X_test, y_test):
    """Fit on the training set, report all metrics on the held-out test set."""
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    m = compute_metrics(cm)
    m["auc"] = float(roc_auc_score(y_test, prob))
    return model, m, cm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    args = ap.parse_args()

    X, y, feature_names = load_data()
    n_mal = int(y.sum())
    print(f"WDBC loaded: {len(y)} samples, {n_mal} malignant, {len(y) - n_mal} benign, "
          f"{X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    print(f"Split: {len(y_train)} train, {len(y_test)} test (stratified)\n")

    os.makedirs(OUT, exist_ok=True)
    models = build_models()
    results = {}

    header = f"{'Model':<22}{'CV acc':>8}{'CV AUC':>8}{'Test acc':>10}{'Sens':>8}{'Spec':>8}{'AUC':>8}"
    print(header)
    print("-" * len(header))

    for name, model in models.items():
        cv_acc, cv_auc = cross_validate(model, X_train, y_train, args.folds)
        fitted, m, cm = evaluate_on_test(model, X_train, y_train, X_test, y_test)
        results[name] = {
            "cv_accuracy": round(cv_acc, 3), "cv_auc": round(cv_auc, 3),
            "test_accuracy": round(m["accuracy"], 3),
            "sensitivity": round(m["sensitivity"], 3),
            "specificity": round(m["specificity"], 3),
            "auc": round(m["auc"], 3),
            "confusion_matrix": {"tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]},
        }
        print(f"{name:<22}{cv_acc:>8.3f}{cv_auc:>8.3f}{m['accuracy']:>10.3f}"
              f"{m['sensitivity']:>8.3f}{m['specificity']:>8.3f}{m['auc']:>8.3f}")

        slug = name.lower().replace(" ", "_")
        joblib.dump(fitted, f"{OUT}/classical_{slug}.joblib")

    # save feature names and a metrics summary for the app and the report
    with open(f"{OUT}/classical_results.json", "w") as f:
        json.dump({"feature_names": feature_names, "results": results}, f, indent=2)

    # identify the best model by sensitivity (the clinically weighted metric),
    # breaking ties on AUC
    best = max(results, key=lambda k: (results[k]["sensitivity"], results[k]["auc"]))
    print(f"\nBest by sensitivity: {best} "
          f"(sensitivity {results[best]['sensitivity']}, AUC {results[best]['auc']})")
    print(f"Saved 5 models and classical_results.json to '{OUT}/'")


if __name__ == "__main__":
    main()
