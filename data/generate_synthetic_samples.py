"""
Generate a small synthetic dataset with the same folder layout prepare_dataset.py
produces, so the rest of the pipeline (train.py, evaluate.py, data.yaml) can be
smoke-tested end-to-end in seconds, with no network access and no GPU required
for the check itself. This is NOT a substitute for training on the real dataset --
it exists purely to catch broken paths/shapes/configs before you spend an hour
on a real GPU run.

Each "aircraft" is a colored rectangle on a noisy background, one shape per class,
placed at a random position/size. Bounding boxes are therefore exact by construction,
which also makes this useful as a correctness check for evaluate.py.

Usage:
    python data/generate_synthetic_samples.py --per-class 8
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CANONICAL_CLASSES as CLASSES

HERE = Path(__file__).resolve().parent
IMG_SIZE = 320


def make_image(cls_id: int, rng: random.Random) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    img = np.full((IMG_SIZE, IMG_SIZE, 3), fill_value=rng.randint(180, 230), dtype=np.uint8)
    noise = (np.random.default_rng(rng.randint(0, 10_000)).normal(0, 8, img.shape)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    w = rng.randint(40, 140)
    h = rng.randint(20, 80)
    x1 = rng.randint(0, IMG_SIZE - w)
    y1 = rng.randint(0, IMG_SIZE - h)
    x2, y2 = x1 + w, y1 + h

    # Deterministic-ish color per class so shapes are visually distinguishable
    color = (
        (cls_id * 37) % 256,
        (cls_id * 91) % 256,
        (cls_id * 149) % 256,
    )
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.putText(img, CLASSES[cls_id], (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cx, cy = (x1 + x2) / 2 / IMG_SIZE, (y1 + y2) / 2 / IMG_SIZE
    bw, bh = w / IMG_SIZE, h / IMG_SIZE
    return img, (cx, cy, bw, bh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=8,
                         help="Synthetic images per class per split multiplier")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    splits = {"train": args.per_class, "val": max(1, args.per_class // 4)}

    for split, n_per_class in splits.items():
        img_dir = HERE / "images" / split
        lbl_dir = HERE / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for cls_id in range(len(CLASSES)):
            for k in range(n_per_class):
                img, (cx, cy, bw, bh) = make_image(cls_id, rng)
                name = f"synthetic_{split}_{cls_id:02d}_{k:03d}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
                (lbl_dir / f"{name}.txt").write_text(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                count += 1
        print(f"{split}: wrote {count} synthetic image/label pairs to {img_dir}")

    print("\nSmoke-test dataset ready. Try:")
    print("  python scripts/train.py --model yolov8n.pt --epochs 1 --batch 8 --imgsz 320")


if __name__ == "__main__":
    main()
