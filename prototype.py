"""
Feature prototype
Baseline convolutional network for benign vs malignant mammogram classification.

This is an exploratory prototype, not the final system. Its purpose is to prove
the hardest and least certain part of the project works end to end: loading
mammogram images, training a convolutional network on an Apple Silicon machine,
and producing a prediction with an honest evaluation.

It runs out of the box on generated (synthetic) data so the whole pipeline can be
demonstrated before the full dataset is downloaded. Point --data_dir at a folder
of real CBIS-DDSM images (arranged in benign/ and malignant/ subfolders) to train
on the real data.

Usage
    python prototype.py                       # synthetic data, quick demo
    python prototype.py --data_dir data       # real images in data/benign, data/malignant
    python prototype.py --data_dir data --regularised   # second ablation tier
    python prototype.py --epochs 15 --img_size 160
"""

import argparse
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from metrics import compute_metrics

SEED = 42
BATCH = 32
OUT = "outputs"

# Fix every source of randomness so a run can be reproduced exactly. Without this
# CPU training drifts between runs and no single result can be pinned, which
# undermines the reproducibility the project claims. enable_op_determinism makes
# the remaining GPU/CPU kernels deterministic too.
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()


def make_synthetic(n, img_size):
    """Create two classes of grey images with a faint, learnable difference.

    Malignant images carry a small bright blob, loosely standing in for a mass,
    so the network has real structure to learn rather than pure noise.
    """
    rng = np.random.default_rng(SEED)
    X = np.zeros((n, img_size, img_size, 1), dtype="float32")
    y = np.zeros((n, 1), dtype="float32")
    for i in range(n):
        label = i % 2
        base = 0.40 if label == 0 else 0.50
        img = rng.normal(base, 0.12, (img_size, img_size, 1)).astype("float32")
        if label == 1:
            cx, cy = rng.integers(25, img_size - 25, size=2)
            r = rng.integers(8, 18)
            yy, xx = np.ogrid[:img_size, :img_size]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
            img[mask, 0] += 0.35
        X[i] = np.clip(img, 0, 1) * 255.0
        y[i] = label
    return X, y


def synthetic_datasets(n, img_size):
    X, y = make_synthetic(n, img_size)
    idx = np.random.default_rng(SEED).permutation(n)
    X, y = X[idx], y[idx]
    n_tr, n_val = int(0.70 * n), int(0.15 * n)
    splits = {
        "train": (X[:n_tr], y[:n_tr]),
        "val":   (X[n_tr:n_tr + n_val], y[n_tr:n_tr + n_val]),
        "test":  (X[n_tr + n_val:], y[n_tr + n_val:]),
    }

    def to_ds(Xa, ya, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((Xa, ya))
        if shuffle:
            ds = ds.shuffle(len(Xa), seed=SEED)
        return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

    return (to_ds(*splits["train"], True),
            to_ds(*splits["val"], False),
            to_ds(*splits["test"], False),
            ["benign", "malignant"])


def directory_datasets(data_dir, img_size):
    common = dict(directory=data_dir, labels="inferred", label_mode="binary",
                  color_mode="grayscale", image_size=(img_size, img_size),
                  batch_size=BATCH, seed=SEED, validation_split=0.30)
    train_ds = keras.utils.image_dataset_from_directory(subset="training", shuffle=True, **common)
    class_names = train_ds.class_names
    hold_ds = keras.utils.image_dataset_from_directory(subset="validation", shuffle=True, **common)
    # Materialise the holdout and shuffle it once, so validation and test each get a
    # mix of both classes rather than being split along class lines.
    Xs, ys = [], []
    for xb, yb in hold_ds:
        Xs.append(xb.numpy()); ys.append(yb.numpy())
    X = np.concatenate(Xs); y = np.concatenate(ys)
    idx = np.random.default_rng(SEED).permutation(len(X))
    X, y = X[idx], y[idx]
    half = len(X) // 2
    auto = tf.data.AUTOTUNE

    def to_ds(Xa, ya):
        return tf.data.Dataset.from_tensor_slices((Xa, ya)).batch(BATCH).prefetch(auto)

    return (train_ds.prefetch(auto), to_ds(X[:half], y[:half]), to_ds(X[half:], y[half:]), class_names)


def build_model(img_size, regularised=False):
    """The convolutional network.

    Baseline (regularised=False): a small CNN with a single dropout layer, the
    first rung of the planned ablation.

    Regularised (regularised=True): the same backbone with on-the-fly image
    augmentation at the input and heavier dropout, the second rung. Augmentation
    also enriches the minority class, which works with the class weighting applied
    during training to counter the imbalance.
    """
    aug = []
    if regularised:
        aug = [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.05, 0.05),
        ]
    dropout_conv = 0.20 if regularised else 0.0
    dropout_head = 0.50 if regularised else 0.30

    blocks = [layers.Input((img_size, img_size, 1)), *aug, layers.Rescaling(1.0 / 255)]
    for filters in (16, 32, 64):
        blocks.append(layers.Conv2D(filters, 3, activation="relu", padding="same"))
        blocks.append(layers.MaxPooling2D())
        if dropout_conv:
            blocks.append(layers.Dropout(dropout_conv))
    blocks += [
        layers.Flatten(),
        layers.Dropout(dropout_head),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
    model = keras.Sequential(blocks)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


def class_weights_from(train_ds):
    """Compute inverse-frequency class weights so the minority (malignant) class
    carries more influence during training. Returns None if a label is absent."""
    labels = np.concatenate([yb.numpy().ravel() for _, yb in train_ds]).astype(int)
    classes = np.unique(labels)
    if len(classes) < 2:
        return None
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def collect_true_pred(model, test_ds):
    y_true = np.concatenate([yb.numpy().ravel() for _, yb in test_ds]) if tf.data.experimental.cardinality(test_ds).numpy() != 0 else np.array([])
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true.astype(int), y_pred, y_prob


def save_plots(history, cm, class_names):
    os.makedirs(OUT, exist_ok=True)
    # training curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/training_curves.png", dpi=130); plt.close(fig)
    # confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], class_names); ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrix.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="folder with benign/ and malignant/ subfolders")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--n", type=int, default=600, help="synthetic sample size")
    ap.add_argument("--regularised", action="store_true",
                    help="second ablation tier: augmentation, heavier dropout, class weighting")
    args = ap.parse_args()

    tier = "regularised" if args.regularised else "baseline"
    print("TensorFlow", tf.__version__)
    print("Tier:", tier)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPU devices visible to TensorFlow:", gpus if gpus else "none (running on CPU)")

    if args.data_dir and os.path.isdir(args.data_dir):
        print(f"\nLoading real images from '{args.data_dir}'")
        train_ds, val_ds, test_ds, class_names = directory_datasets(args.data_dir, args.img_size)
    else:
        if args.data_dir:
            print(f"\n'{args.data_dir}' not found, falling back to synthetic data.")
        else:
            print("\nNo --data_dir given, using synthetic data to demonstrate the pipeline.")
        train_ds, val_ds, test_ds, class_names = synthetic_datasets(args.n, args.img_size)

    print("Classes:", class_names)
    model = build_model(args.img_size, regularised=args.regularised)
    model.summary()

    fit_kwargs = dict(validation_data=val_ds, epochs=args.epochs, verbose=2)
    if args.regularised:
        cw = class_weights_from(train_ds)
        if cw:
            print("Class weights:", cw)
            fit_kwargs["class_weight"] = cw

    print("\nTraining...")
    history = model.fit(train_ds, **fit_kwargs)

    print("\nEvaluating on the held-out test set...")
    loss, acc, auc = model.evaluate(test_ds, verbose=0)
    y_true, y_pred, _ = collect_true_pred(model, test_ds)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    m = compute_metrics(cm)
    sensitivity = m["sensitivity"]
    specificity = m["specificity"]

    print("\n================ RESULTS ================")
    print(f"Tier              : {tier}")
    print(f"Test accuracy     : {acc:.3f}")
    print(f"Test AUC          : {auc:.3f}")
    print(f"Sensitivity (recall, malignant): {sensitivity:.3f}")
    print(f"Specificity                    : {specificity:.3f}")
    print("\nConfusion matrix [rows = actual, cols = predicted]:")
    print(f"            pred {class_names[0]:>9}  pred {class_names[1]:>9}")
    print(f"actual {class_names[0]:>9}   {tn:>9}      {fp:>9}")
    print(f"actual {class_names[1]:>9}   {fn:>9}      {tp:>9}")
    print("\n" + classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    save_plots(history, cm, class_names)
    os.makedirs(OUT, exist_ok=True)
    model_name = "regularised_cnn.keras" if args.regularised else "baseline_cnn.keras"
    model.save(f"{OUT}/{model_name}")
    print(f"Saved model and plots to '{OUT}/'")


if __name__ == "__main__":
    main()