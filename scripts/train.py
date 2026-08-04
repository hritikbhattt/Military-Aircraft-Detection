"""
Train a YOLOv8 model on the 103-class military aircraft dataset.

Usage:
    python scripts/train.py
    python scripts/train.py --model yolov8m.pt --epochs 100 --batch 16 --imgsz 640
    python scripts/train.py --resume runs/detect/train/weights/last.pt
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "data.yaml"
MODELS_DIR = ROOT / "models"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8s.pt",
                         help="Base checkpoint to fine-tune: yolov8n/s/m/l/x.pt")
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0, help="GPU index, list ('0,1'), or 'cpu'")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", default=True,
                         help="Mixed precision training (default on)")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    parser.add_argument("--name", default="train")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"{args.data} not found or its images/labels folders are empty.\n"
            "Run `python data/prepare_dataset.py` first."
        )

    model = YOLO(args.resume if args.resume else args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        amp=args.amp,
        project=args.project,
        name=args.name,
        resume=bool(args.resume),
        plots=True,
    )

    # Promote the best checkpoint from this run into models/best.pt for detect.py
    best_ckpt = Path(results.save_dir) / "weights" / "best.pt"
    if best_ckpt.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        shutil.copy2(best_ckpt, MODELS_DIR / "best.pt")
        print(f"\nBest checkpoint copied to {MODELS_DIR / 'best.pt'}")

    # Quick validation summary on the held-out split
    metrics = model.val(data=args.data)
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
