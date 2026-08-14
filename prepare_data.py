"""
prepare_data.py

Sorts CBIS-DDSM (Kaggle JPEG version) into the benign/ and malignant/ folders that
prototype.py expects. The pathology label for each image is NOT in the folder names,
it is in the description CSV files, so this script reads those CSVs, finds the matching
image folder under jpeg/, and copies a sample into the right class folder.

Run it like this (point --root at the folder that CONTAINS the csv/ and jpeg/ folders):

    python prepare_data.py --root /path/to/CBIS-DDSM --out data --per_class 200

Then train:

    python prototype.py --data_dir data --epochs 15

If it copies nothing, run it again and paste me the diagnostic lines it prints. The
exact column names in this dataset have changed between versions and I can adjust the
matching in one line.
"""

import argparse
import os
import glob
import shutil
import random
import pandas as pd

random.seed(42)


def find_dir(root, name):
    """Locate a subfolder called `name` at the root or one level down."""
    direct = os.path.join(root, name)
    if os.path.isdir(direct):
        return direct
    hits = glob.glob(os.path.join(root, "*", name))
    return hits[0] if hits else None


def pick_column(df, *candidates):
    """Return the real column name matching one of the candidate names, ignoring
    case and surrounding spaces."""
    norm = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    # looser fallback: first column containing all the words of the first candidate
    words = candidates[0].split()
    for low, real in norm.items():
        if all(w in low for w in words):
            return real
    return None


def label_of(pathology):
    p = str(pathology).strip().upper()
    if p.startswith("MALIGNANT"):
        return "malignant"
    if p.startswith("BENIGN"):
        return "benign"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing csv/ and jpeg/")
    ap.add_argument("--out", default="data")
    ap.add_argument("--per_class", type=int, default=200)
    args = ap.parse_args()

    csv_dir = find_dir(args.root, "csv")
    jpeg_dir = find_dir(args.root, "jpeg")
    print(f"csv folder : {csv_dir}")
    print(f"jpeg folder: {jpeg_dir}")
    if not csv_dir or not jpeg_dir:
        print("Could not find csv/ and jpeg/ under --root. Check the path.")
        return

    existing = set(os.listdir(jpeg_dir))
    print(f"jpeg subfolders found: {len(existing)}")

    description_csvs = [f for f in glob.glob(os.path.join(csv_dir, "*.csv"))
                        if "case_description" in os.path.basename(f).lower()]
    print(f"description CSVs: {[os.path.basename(f) for f in description_csvs]}")

    # gather (label, image_file_path) pairs from every description CSV
    rows = []
    for f in description_csvs:
        df = pd.read_csv(f)
        path_col = pick_column(df, "image file path", "image_file_path")
        path_col = path_col or pick_column(df, "image path")
        patho_col = pick_column(df, "pathology")
        if not path_col or not patho_col:
            print(f"  skipping {os.path.basename(f)}: columns not found "
                  f"(saw {list(df.columns)[:6]} ...)")
            continue
        for _, r in df.iterrows():
            lab = label_of(r[patho_col])
            if lab:
                rows.append((lab, str(r[path_col])))
    print(f"labelled rows collected: {len(rows)}")
    random.shuffle(rows)

    for cls in ("benign", "malignant"):
        os.makedirs(os.path.join(args.out, cls), exist_ok=True)

    counts = {"benign": 0, "malignant": 0}
    misses = 0
    for lab, path in rows:
        if counts[lab] >= args.per_class:
            if all(counts[c] >= args.per_class for c in counts):
                break
            continue
        # find which path component is an actual jpeg subfolder
        folder = next((c for c in path.replace("\\", "/").split("/") if c in existing), None)
        if folder is None:
            misses += 1
            continue
        imgs = glob.glob(os.path.join(jpeg_dir, folder, "*.jpg")) + \
               glob.glob(os.path.join(jpeg_dir, folder, "*.jpeg")) + \
               glob.glob(os.path.join(jpeg_dir, folder, "*.png"))
        if not imgs:
            misses += 1
            continue
        src = imgs[0]
        dst = os.path.join(args.out, lab, f"{folder}_{os.path.basename(src)}")
        shutil.copy2(src, dst)
        counts[lab] += 1

    print(f"\nCopied: benign={counts['benign']}  malignant={counts['malignant']}")
    if misses:
        print(f"Rows with no matching jpeg folder: {misses}")
    if counts["benign"] == 0 and counts["malignant"] == 0:
        print("\nNothing matched. Paste the diagnostic lines above to me and I will "
              "adjust the column or path matching.")
    else:
        print(f"\nDone. Now run:  python prototype.py --data_dir {args.out} --epochs 15")


if __name__ == "__main__":
    main()
