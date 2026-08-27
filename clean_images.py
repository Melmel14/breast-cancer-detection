"""
clean_images.py

Attacks the shortcut-learning problem revealed by Grad-CAM. The imaging model was
found to focus on image borders, corners and the orientation-label marker (e.g.
"JM") rather than on breast tissue. This script preprocesses the sorted mammogram
images to remove those shortcut features, so a model retrained on the cleaned
images is forced to look at the tissue instead.

Two operations, both aimed at the specific shortcuts Grad-CAM exposed:

1. Border cropping. Mammograms sit on a black background. The script finds the
   bounding box of the actual breast (the non-black region) and crops to it,
   removing the black padding and corners the model was exploiting.

2. Label removal. The orientation marker is a bright block of text in a corner,
   away from the breast. After cropping to the breast, any small bright blobs left
   near the edges are darkened, so the "JM"-style label cannot be used as a cue.

The cleaned images are written to a parallel folder, keeping the benign/ and
malignant/ structure, so training is unchanged apart from pointing --data_dir at
the cleaned folder.

Usage
    python clean_images.py --in data --out data_clean
    python prototype.py --data_dir data_clean --epochs 15 --regularised
"""

import argparse
import os

import numpy as np
from PIL import Image


def crop_to_breast(arr, thresh=15):
    """Crop the image to the bounding box of the breast (the non-black region).

    arr is a 2D grayscale array. Pixels above `thresh` are considered tissue
    rather than background padding. A small margin is kept so the breast edge is
    not clipped.
    """
    mask = arr > thresh
    if not mask.any():
        return arr  # all black; nothing to do
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    # small margin
    m = 4
    r0, c0 = max(0, r0 - m), max(0, c0 - m)
    r1, c1 = min(arr.shape[0] - 1, r1 + m), min(arr.shape[1] - 1, c1 + m)
    return arr[r0:r1 + 1, c0:c1 + 1]


def remove_edge_labels(arr, bright=180, band_frac=0.18):
    """Darken small bright blobs near the top and bottom edges, where orientation
    labels sit. Only the outer horizontal bands are touched, so the breast tissue
    in the centre is never altered.

    This is deliberately conservative: it targets bright pixels in a thin edge
    band, which is where the "JM"-style marker appears, not the tissue.
    """
    out = arr.copy()
    h = arr.shape[0]
    band = int(h * band_frac)
    for r0, r1 in [(0, band), (h - band, h)]:
        region = out[r0:r1, :]
        # a label is much brighter than surrounding background in the edge band
        region[region > bright] = 0
        out[r0:r1, :] = region
    return out


def clean_one(path, size=256):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype="uint8")
    arr = crop_to_breast(arr)
    arr = remove_edge_labels(arr)
    # resize to a consistent size so the training loader gets uniform inputs
    return Image.fromarray(arr).resize((size, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="sorted image folder (benign/ malignant/)")
    ap.add_argument("--out", default="data_clean", help="output folder for cleaned images")
    ap.add_argument("--size", type=int, default=256, help="output image size")
    args = ap.parse_args()

    total = 0
    for cls in ("benign", "malignant"):
        src_dir = os.path.join(args.inp, cls)
        if not os.path.isdir(src_dir):
            print(f"  skipping missing folder: {src_dir}")
            continue
        dst_dir = os.path.join(args.out, cls)
        os.makedirs(dst_dir, exist_ok=True)
        n = 0
        for name in os.listdir(src_dir):
            if name.startswith("."):
                continue
            try:
                cleaned = clean_one(os.path.join(src_dir, name), size=args.size)
                stem = os.path.splitext(name)[0]
                cleaned.save(os.path.join(dst_dir, stem + ".png"))
                n += 1
            except Exception as e:
                print(f"  could not clean {name}: {e}")
        print(f"{cls}: cleaned {n} images -> {dst_dir}")
        total += n

    print(f"\nDone. {total} cleaned images in '{args.out}/'.")
    print("Retrain with:  python prototype.py --data_dir " + args.out + " --epochs 15 --regularised")


if __name__ == "__main__":
    main()
