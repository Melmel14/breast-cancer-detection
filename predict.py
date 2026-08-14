"""
predict.py

Run the trained baseline CNN on a single mammogram image and print a benign or
malignant result with a confidence level. This is the core of the final product in its
simplest form: an image goes in, a result comes out. The web interface in your design
will wrap this exact step later, during development.

First train a model so it exists on disk:

    python prototype.py --data_dir data --epochs 15      # or synthetic: python prototype.py

Then classify any single image:

    python predict.py --image path/to/mammogram.jpg
    python predict.py --image x.png --model outputs/baseline_cnn.keras
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

CLASS_NAMES = ["benign", "malignant"]  # alphabetical, matches training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to one image")
    ap.add_argument("--model", default="outputs/baseline_cnn.keras")
    args = ap.parse_args()

    model = keras.models.load_model(args.model)
    _, h, w, _ = model.input_shape  # the size the model was trained on

    img = keras.utils.load_img(args.image, color_mode="grayscale", target_size=(h, w))
    arr = keras.utils.img_to_array(img)          # shape (h, w, 1), values 0 to 255
    arr = np.expand_dims(arr, axis=0)            # add the batch dimension

    prob_malignant = float(model.predict(arr, verbose=0).ravel()[0])
    if prob_malignant >= 0.5:
        label, confidence = "malignant", prob_malignant
    else:
        label, confidence = "benign", 1.0 - prob_malignant

    print(f"\nImage      : {args.image}")
    print(f"Prediction : {label.upper()}")
    print(f"Confidence : {confidence * 100:.1f}%")
    print(f"(raw model output, probability of malignant = {prob_malignant:.3f})")
    print("\nReminder: this is decision support, not a diagnosis. A clinician decides.")


if __name__ == "__main__":
    main()
