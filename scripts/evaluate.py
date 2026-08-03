"""
Evaluate a trained checkpoint: official ultralytics mAP/precision/recall,
plus a custom per-class AP table and a confusion matrix, saved to outputs/eval/.

Usage:
    python scripts/evaluate.py --weights models/best.pt --data data/data.yaml
    python scripts/evaluate.py --weights models/best.pt --split test
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from metrics import evaluate_dataset  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "eval"


def run_ultralytics_val(weights: str, data: str, split: str):
    from ultralytics import YOLO
    model = YOLO(weights)
    metrics = model.val(data=data, split=split)
    print(f"\n[ultralytics] mAP50:    {metrics.box.map50:.4f}")
    print(f"[ultralytics] mAP50-95: {metrics.box.map:.4f}")
    print(f"[ultralytics] Precision:{metrics.box.mp:.4f}  Recall: {metrics.box.mr:.4f}")
    return metrics


def predict_labels_for_split(weights: str, images_dir: Path, out_dir: Path, conf: float):
    """Run the model over a split's images and dump YOLO-format prediction txts (with conf)."""
    from ultralytics import YOLO
    model = YOLO(weights)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for img_path in img_paths:
        result = model.predict(str(img_path), conf=conf, verbose=False)[0]
        lines = []
        for box in result.boxes:
            cls_id = int(box.cls)
            cx, cy, w, h = box.xywhn[0].tolist()
            conf_val = float(box.conf)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf_val:.6f}")
        (out_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return img_paths


def plot_per_class_ap(report: dict, class_names: list[str], out_path: Path):
    names = [c for c in class_names if c in report]
    aps = [report[c]["ap"] for c in names]
    order = np.argsort(aps)
    names = [names[i] for i in order]
    aps = [aps[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    ax.barh(names, aps, color="#3b6ea5")
    ax.set_xlabel("AP@0.5")
    ax.set_xlim(0, 1)
    ax.set_title("Per-class Average Precision")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    parser.add_argument("--data", default=str(ROOT / "data" / "data.yaml"))
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--skip-ultralytics", action="store_true",
                         help="Skip the official ultralytics val() call (e.g. if torch isn't installed) "
                              "and only run the custom per-class report against existing predictions.")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists() and not args.skip_ultralytics:
        sys.exit(f"No weights found at {weights_path}. Train first with scripts/train.py.")

    data_cfg = yaml.safe_load(Path(args.data).read_text())
    class_names = data_cfg["names"]
    data_root = Path(args.data).resolve().parent

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_ultralytics:
        run_ultralytics_val(args.weights, args.data, args.split)

        images_dir = data_root / data_cfg[args.split]
        pred_dir = OUTPUT_DIR / "predictions"
        print(f"\nRunning inference over {args.split} split for the custom per-class report...")
        predict_labels_for_split(args.weights, images_dir, pred_dir, args.conf)

        gt_dir = data_root / "labels" / args.split
        report = evaluate_dataset(str(gt_dir), str(pred_dir), class_names, iou_thres=args.iou_thres)

        print(f"\n{'Class':16s}{'AP@0.5':>8s}{'Prec':>8s}{'Recall':>8s}{'GT':>6s}{'Pred':>6s}")
        for name in class_names:
            if name not in report:
                continue
            r = report[name]
            print(f"{name:16s}{r['ap']:8.3f}{r['precision']:8.3f}{r['recall']:8.3f}{r['n_gt']:6d}{r['n_pred']:6d}")
        print(f"\nCustom mAP50 ({args.split}): {report['_summary']['mAP50']:.4f} "
              f"over {report['_summary']['n_images']} images, "
              f"{report['_summary']['n_classes_evaluated']} classes present.")

        plot_path = OUTPUT_DIR / "per_class_ap.png"
        plot_per_class_ap(report, class_names, plot_path)
        print(f"\nSaved per-class AP chart to {plot_path}")
        print("Ultralytics also saves its own confusion matrix / PR curves under runs/detect/val*/")


if __name__ == "__main__":
    main()
