"""
Download the "Military Aircraft Detection Dataset" (Kaggle, a2015003713) and
convert it from its native per-image CSV annotations (VOC-style pixel boxes:
filename, width, height, class, xmin, ymin, xmax, ymax) into YOLO txt format,
split into train/val/test, and write it under data/images/ + data/labels/.

Requires a Kaggle account. Run once, locally or on Kaggle/Colab (needs network
access, which this repo does not assume you have in every environment):

    python data/prepare_dataset.py --limit-per-class 150

Auth: put your kaggle.json in ~/.kaggle/kaggle.json, or set
KAGGLE_USERNAME / KAGGLE_KEY env vars. See https://www.kaggle.com/docs/api
"""
import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from classes import CLASSES, class_index, normalize_raw_label

DATASET_SLUG = "a2015003713/militaryaircraftdetectiondataset"

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
IMAGES_DIR = HERE / "images"
LABELS_DIR = HERE / "labels"


def download_raw() -> Path:
    """Download the raw Kaggle dataset via kagglehub and return its local path."""
    import kagglehub

    print(f"Downloading '{DATASET_SLUG}' from Kaggle (cached after first run)...")
    path = kagglehub.dataset_download(DATASET_SLUG)
    return Path(path)


def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert a pixel-space VOC box to normalized YOLO (cx, cy, w, h)."""
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h


def collect_annotations(raw_dir: Path, limit_per_class: int | None):
    """
    Walk the raw dataset for image/csv pairs, keep only boxes whose class is
    in our 36-class list, and group YOLO-format lines by source image.
    Returns: dict[image_path] -> list[str] (YOLO label lines), and per-class counts.
    """
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(
            f"No CSV annotation files found under {raw_dir}. "
            "The dataset layout may have changed -- inspect it manually."
        )

    per_image_lines: dict[Path, list[str]] = defaultdict(list)
    per_class_images: dict[str, set[Path]] = defaultdict(set)

    for csv_path in csv_files:
        img_path = csv_path.with_suffix(".jpg")
        if not img_path.exists():
            img_path = csv_path.with_suffix(".png")
            if not img_path.exists():
                continue

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_class = row.get("class") or row.get("type")
                cls_name = normalize_raw_label(raw_class or "")
                if cls_name is None:
                    continue  # not one of our 36 target classes

                img_w, img_h = int(row["width"]), int(row["height"])
                xmin, ymin = float(row["xmin"]), float(row["ymin"])
                xmax, ymax = float(row["xmax"]), float(row["ymax"])
                cx, cy, w, h = voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)

                cls_id = class_index(cls_name)
                per_image_lines[img_path].append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                per_class_images[cls_name].add(img_path)

    if limit_per_class:
        keep: set[Path] = set()
        for cls_name, imgs in per_class_images.items():
            imgs = sorted(imgs)
            random.Random(42).shuffle(imgs)
            keep.update(imgs[:limit_per_class])
        per_image_lines = {p: lines for p, lines in per_image_lines.items() if p in keep}

    counts = {c: len(imgs) for c, imgs in per_class_images.items()}
    return per_image_lines, counts


def write_split(per_image_lines: dict, splits=(0.8, 0.1, 0.1), seed=42):
    images = sorted(per_image_lines.keys())
    random.Random(seed).shuffle(images)

    n = len(images)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    split_map = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split, imgs in split_map.items():
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            dst_img = IMAGES_DIR / split / img_path.name
            dst_lbl = LABELS_DIR / split / (img_path.stem + ".txt")
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(per_image_lines[img_path]) + "\n")

    return {split: len(imgs) for split, imgs in split_map.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-class", type=int, default=None,
                         help="Cap images per class (useful for a quick smoke run).")
    parser.add_argument("--raw-dir", type=str, default=None,
                         help="Skip download; point directly at an already-downloaded dataset dir.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else download_raw()

    print("Parsing CSV annotations and converting to YOLO format...")
    per_image_lines, class_counts = collect_annotations(raw_dir, args.limit_per_class)

    if not per_image_lines:
        raise RuntimeError("No matching images found for the 36 target classes -- check classes.py mappings.")

    print(f"Matched {len(per_image_lines)} images across {len(class_counts)} classes:")
    for c in CLASSES:
        print(f"  {c:14s} {class_counts.get(c, 0)}")

    split_counts = write_split(per_image_lines)
    print(f"\nDone. Split sizes: {split_counts}")
    print(f"Images:  {IMAGES_DIR}")
    print(f"Labels:  {LABELS_DIR}")
    print("\nNext: python scripts/train.py")


if __name__ == "__main__":
    main()
