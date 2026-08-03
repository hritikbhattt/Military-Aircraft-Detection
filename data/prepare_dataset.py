"""
Download the "Military Aircraft Detection Dataset" (Kaggle, a2015003713) and
convert it from labels_with_split.csv (VOC-style pixel boxes: filename,
width, height, class, xmin, ymin, xmax, ymax, split) into YOLO txt format,
using the dataset's own train/validation/test split, written under
data/images/ + data/labels/.

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

from classes import CANONICAL_CLASSES, CLASS_TO_INDEX, normalize_label

DATASET_SLUG = "a2015003713/militaryaircraftdetectiondataset"

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "images"
LABELS_DIR = HERE / "labels"

# The dataset's own split names -> the folder names we want locally.
SPLIT_NAME_MAP = {"train": "train", "validation": "validation", "test": "test"}


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


def find_labels_csv(raw_dir: Path) -> Path:
    matches = list(raw_dir.rglob("labels_with_split.csv"))
    if not matches:
        raise RuntimeError(
            f"Could not find labels_with_split.csv under {raw_dir}. "
            "The dataset layout may have changed -- inspect it manually."
        )
    return matches[0]


def find_dataset_images_dir(raw_dir: Path) -> Path:
    """The folder containing the actual .jpg files (same folder as the
    per-image CSVs, named 'dataset')."""
    matches = [p for p in raw_dir.rglob("dataset") if p.is_dir()]
    if not matches:
        raise RuntimeError(f"Could not find a 'dataset' image folder under {raw_dir}.")
    return matches[0]


def collect_annotations(raw_dir: Path, limit_per_class: int | None):
    """
    Read labels_with_split.csv, keep only rows whose class is in our
    canonical class list, and group YOLO-format lines by (image path, split).
    Returns: dict[(split, image_path)] -> list[str] (YOLO label lines),
             and per-class counts.
    """
    labels_csv = find_labels_csv(raw_dir)
    images_dir = find_dataset_images_dir(raw_dir)

    per_image_lines: dict[tuple[str, Path], list[str]] = defaultdict(list)
    per_class_images: dict[str, set[Path]] = defaultdict(set)

    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_name = normalize_label(row["class"])
            if cls_name not in CLASS_TO_INDEX:
                continue  # not one of our target classes

            split = SPLIT_NAME_MAP.get(row["split"].strip().lower())
            if split is None:
                continue  # unknown split value, skip defensively

            img_path = images_dir / f"{row['filename']}.jpg"
            if not img_path.exists():
                continue

            img_w, img_h = int(row["width"]), int(row["height"])
            xmin, ymin = float(row["xmin"]), float(row["ymin"])
            xmax, ymax = float(row["xmax"]), float(row["ymax"])
            cx, cy, w, h = voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)

            cls_id = CLASS_TO_INDEX[cls_name]
            key = (split, img_path)
            per_image_lines[key].append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            per_class_images[cls_name].add(img_path)

    if limit_per_class:
        keep: set[Path] = set()
        for cls_name, imgs in per_class_images.items():
            imgs = sorted(imgs)
            random.Random(42).shuffle(imgs)
            keep.update(imgs[:limit_per_class])
        per_image_lines = {
            key: lines for key, lines in per_image_lines.items() if key[1] in keep
        }

    counts = {c: len(imgs) for c, imgs in per_class_images.items()}
    return per_image_lines, counts


def write_split(per_image_lines: dict):
    split_counts: dict[str, int] = defaultdict(int)

    for (split, img_path), lines in per_image_lines.items():
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

        dst_img = IMAGES_DIR / split / img_path.name
        dst_lbl = LABELS_DIR / split / (img_path.stem + ".txt")
        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(lines) + "\n")
        split_counts[split] += 1

    return dict(split_counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-class", type=int, default=None,
                         help="Cap images per class (useful for a quick smoke run).")
    parser.add_argument("--raw-dir", type=str, default=None,
                         help="Skip download; point directly at an already-downloaded dataset dir.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else download_raw()

    print("Parsing labels_with_split.csv and converting to YOLO format...")
    per_image_lines, class_counts = collect_annotations(raw_dir, args.limit_per_class)

    if not per_image_lines:
        raise RuntimeError("No matching images found -- check classes.py mappings and dataset paths.")

    print(f"Matched {len(per_image_lines)} (image, split) pairs across {len(class_counts)} classes:")
    for c in CANONICAL_CLASSES:
        print(f"  {c:14s} {class_counts.get(c, 0)}")

    split_counts = write_split(per_image_lines)
    print(f"\nDone. Split sizes: {split_counts}")
    print(f"Images:  {IMAGES_DIR}")
    print(f"Labels:  {LABELS_DIR}")
    print("\nNext: python scripts/train.py")


if __name__ == "__main__":
    main()