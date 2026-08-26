"""
prepare_mias.py

Prepares the mini-MIAS mammography database for use as an EXTERNAL VALIDATION set
for the imaging model. Mini-MIAS is a different source from CBIS-DDSM (older UK
film scans, different scanner and population), so testing the trained model on it
measures whether the model generalises beyond the data it was trained on, which is
a gap the literature review identified.

Mini-MIAS labels live in Info.txt, one row per abnormality:

    REFNUM BG CLASS SEVERITY X Y RADIUS
    mdb001 G CIRC B 535 425 197

SEVERITY is B (benign) or M (malignant); NORM rows have no severity and are
skipped, because the model was trained on a benign-vs-malignant task rather than
normal-vs-abnormal. Some images appear on more than one row (several
abnormalities); each image is used once, and if any row marks it malignant the
image is treated as malignant, since a missed malignancy is the costlier error.

The images are .pgm (portable grey map); they are converted to grayscale .png and
sorted into benign/ and malignant/ folders, the structure the model expects.

Usage
    python prepare_mias.py --root archive --out mias_data
"""

import argparse
import os

from PIL import Image


def read_labels(info_path):
    """Return {refnum: 'benign'|'malignant'} from Info.txt.

    If an image has both a benign and a malignant abnormality across rows, it is
    labelled malignant (the safety-conservative choice).
    """
    labels = {}
    with open(info_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            ref = parts[0]
            if ref.upper() == "REFNUM":  # header line
                continue
            # severity is the 4th column when present; NORM rows are shorter
            severity = parts[3] if len(parts) >= 4 else None
            if severity == "M":
                labels[ref] = "malignant"
            elif severity == "B":
                # do not downgrade an image already seen as malignant
                labels.setdefault(ref, "benign")
            # NORM / anything else: skip
    return labels


def find_pgm(root, ref):
    """Locate the .pgm for a reference number, tolerating layout differences."""
    candidates = [
        os.path.join(root, ref + ".pgm"),
        os.path.join(root, "all-mias", ref + ".pgm"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fall back to a shallow search
    for dirpath, _, files in os.walk(root):
        if ref + ".pgm" in files:
            return os.path.join(dirpath, ref + ".pgm")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing Info.txt and the .pgm files")
    ap.add_argument("--out", default="mias_data", help="output folder for sorted images")
    args = ap.parse_args()

    info_path = None
    for name in ("Info.txt", "info.txt", "Info.TXT"):
        p = os.path.join(args.root, name)
        if os.path.exists(p):
            info_path = p
            break
    if info_path is None:
        print(f"Could not find Info.txt under '{args.root}'. Check the --root path.")
        return

    labels = read_labels(info_path)
    n_ben = sum(1 for v in labels.values() if v == "benign")
    n_mal = sum(1 for v in labels.values() if v == "malignant")
    print(f"Info.txt: {len(labels)} abnormal images ({n_ben} benign, {n_mal} malignant); "
          f"normal images skipped.")

    for cls in ("benign", "malignant"):
        os.makedirs(os.path.join(args.out, cls), exist_ok=True)

    converted, missing = 0, 0
    for ref, cls in labels.items():
        src = find_pgm(args.root, ref)
        if src is None:
            missing += 1
            continue
        try:
            img = Image.open(src).convert("L")  # grayscale
            dst = os.path.join(args.out, cls, ref + ".png")
            img.save(dst)
            converted += 1
        except Exception as e:
            print(f"  could not convert {ref}: {e}")
            missing += 1

    print(f"Converted {converted} images into '{args.out}/benign' and '{args.out}/malignant'.")
    if missing:
        print(f"{missing} images could not be found or converted (see messages above).")


if __name__ == "__main__":
    main()
