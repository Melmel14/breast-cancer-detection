"""
evaluate_external.py

External validation of a trained imaging model on the mini-MIAS dataset, a
different source from the CBIS-DDSM data the model was trained on. This measures
whether the model generalises beyond its training distribution, which the
literature review identified as a common gap.

Run prepare_mias.py first to produce the sorted folder, then:

    python evaluate_external.py --data_dir mias_data
    python evaluate_external.py --data_dir mias_data --model outputs/regularised_cnn.keras

By default it evaluates the regularised model, which is the best imaging model.
The result is expected to be lower than on CBIS-DDSM, because this is a genuinely
different source; a drop is a legitimate and reportable finding, not a failure.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score

from metrics import compute_metrics

IMG_SIZE = 128
BATCH = 32
CLASS_NAMES = ["benign", "malignant"]


def load_dataset(data_dir):
    """Load all sorted images as one evaluation set (no training split needed;
    the model is already trained, this is pure external testing)."""
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, labels="inferred", label_mode="binary",
        color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH, shuffle=False,
    )
    class_names = ds.class_names
    return ds, class_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="mias_data", help="sorted mini-MIAS folder")
    ap.add_argument("--model", default="outputs/regularised_cnn.keras",
                    help="saved model to evaluate")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"No folder '{args.data_dir}'. Run prepare_mias.py first.")
        return
    if not os.path.exists(args.model):
        print(f"No model at '{args.model}'. Train it first, or pass --model.")
        return

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    ds, class_names = load_dataset(args.data_dir)
    print(f"External set: classes {class_names}")

    y_true = np.concatenate([y.numpy().ravel() for _, y in ds]).astype(int)
    y_prob = model.predict(ds, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    m = compute_metrics(cm)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # only one class present

    tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
    print("\n========== EXTERNAL VALIDATION (mini-MIAS) ==========")
    print(f"Model             : {os.path.basename(args.model)}")
    print(f"Images evaluated  : {len(y_true)} ({int(y_true.sum())} malignant, "
          f"{int((1 - y_true).sum())} benign)")
    print(f"Accuracy          : {m['accuracy']:.3f}")
    print(f"AUC               : {auc:.3f}")
    print(f"Sensitivity       : {m['sensitivity']:.3f}")
    print(f"Specificity       : {m['specificity']:.3f}")
    print("\nConfusion matrix [rows = actual, cols = predicted]:")
    print(f"            pred {class_names[0]:>9}  pred {class_names[1]:>9}")
    print(f"actual {class_names[0]:>9}   {tn:>9}      {fp:>9}")
    print(f"actual {class_names[1]:>9}   {fn:>9}      {tp:>9}")
    print("\nNote: mini-MIAS is a different source from the CBIS-DDSM training data,")
    print("so a lower score here reflects limited generalisation and is a")
    print("legitimate external-validation finding.")


if __name__ == "__main__":
    main()
