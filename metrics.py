"""
metrics.py

Clinical evaluation metrics for the imaging pathway.

This logic used to live inline inside prototype.py's main(), which meant it could
only be exercised by running a full training job. Pulling it out makes the most
important arithmetic in the project testable in milliseconds, with no model, no
images and no TensorFlow import.

Convention throughout: class 0 is benign, class 1 is malignant, so a false
negative is a malignant case reported as benign, which is the error that matters
most in this setting.
"""


def compute_metrics(cm):
    """Derive accuracy, sensitivity and specificity from a 2x2 confusion matrix.

    Args:
        cm: a 2x2 confusion matrix, rows actual, columns predicted, ordered
            [benign, malignant]. Accepts a numpy array or a nested sequence.

    Returns:
        dict with keys accuracy, sensitivity, specificity, and the raw counts
        tn, fp, fn, tp.

    Each rate is guarded against a zero denominator, which can occur on a small
    or degenerate split where one class is absent from the test set. Returning
    0.0 rather than raising keeps a training run from dying at the final step.
    """
    (tn, fp), (fn, tp) = cm

    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    total = tn + fp + fn + tp

    accuracy = (tn + tp) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
