"""
Pure-numpy detection metrics: IoU, per-class AP (VOC-style 101-point interpolation),
and mAP -- deliberately dependency-light (numpy only) so it can be unit-tested without
torch/ultralytics installed, and reused by scripts/evaluate.py for a custom per-class
report on top of whatever ultralytics itself reports.

Ground truth / prediction file format (YOLO-style, one file per image):
    gt:   "<class_id> <cx> <cy> <w> <h>"                (normalized 0-1)
    pred: "<class_id> <cx> <cy> <w> <h> <confidence>"    (normalized 0-1)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def xywhn_to_xyxy(box: np.ndarray) -> np.ndarray:
    """[cx, cy, w, h] (normalized) -> [x1, y1, x2, y2] (normalized)."""
    cx, cy, w, h = box
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _read_yolo_file(path: Path, has_conf: bool) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = [float(x) for x in line.split()]
        cls_id = int(parts[0])
        box = np.array(parts[1:5])
        conf = parts[5] if has_conf and len(parts) > 5 else 1.0
        rows.append({"cls": cls_id, "box": xywhn_to_xyxy(box), "conf": conf})
    return rows


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    VOC-style 101-point interpolated average precision from PR points.

    precision(r) is defined as the max precision among all points with
    recall' >= r. We look this up via searchsorted rather than np.interp,
    because mrec can contain duplicate x-values (e.g. a single data point
    at recall=1.0 plus the appended sentinel) which makes linear
    interpolation between them ambiguous/wrong at the boundary.
    """
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    recall_levels = np.linspace(0, 1, 101)
    precisions_at_levels = []
    for r in recall_levels:
        idx = np.searchsorted(mrec, r, side="left")
        idx = min(idx, len(mpre) - 1)
        precisions_at_levels.append(mpre[idx])
    return float(np.mean(precisions_at_levels))


def evaluate_class(gts: list[dict], preds: list[dict], iou_thres: float = 0.5) -> dict:
    """
    gts/preds: list of {"image_id": ..., "box": xyxy, "conf": float} already
    filtered to a single class.
    Returns precision/recall arrays and AP for that class.
    """
    preds_sorted = sorted(preds, key=lambda p: -p["conf"])
    n_gt = len(gts)
    matched = {i: False for i in range(len(gts))}

    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))

    for i, pred in enumerate(preds_sorted):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if gt["image_id"] != pred["image_id"] or matched[j]:
                continue
            iou = iou_xyxy(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thres and best_j >= 0:
            tp[i] = 1
            matched[best_j] = True
        else:
            fp[i] = 1

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / n_gt if n_gt > 0 else np.zeros_like(tp_cum)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = compute_ap(recall, precision)

    return {
        "ap": ap,
        "precision": float(precision[-1]) if len(precision) else 0.0,
        "recall": float(recall[-1]) if len(recall) else 0.0,
        "n_gt": n_gt,
        "n_pred": len(preds_sorted),
    }


def evaluate_dataset(gt_dir: str, pred_dir: str, class_names: list[str], iou_thres: float = 0.5) -> dict:
    """
    Walk matching *.txt files in gt_dir / pred_dir (same stem = same image) and
    compute per-class AP/precision/recall plus overall mAP.
    """
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    gt_files = sorted(gt_dir.glob("*.txt"))

    per_class_gt: dict[int, list[dict]] = {i: [] for i in range(len(class_names))}
    per_class_pred: dict[int, list[dict]] = {i: [] for i in range(len(class_names))}

    image_ids = set()
    for gt_file in gt_files:
        image_id = gt_file.stem
        image_ids.add(image_id)
        for row in _read_yolo_file(gt_file, has_conf=False):
            per_class_gt[row["cls"]].append({"image_id": image_id, "box": row["box"]})

        pred_file = pred_dir / gt_file.name
        for row in _read_yolo_file(pred_file, has_conf=True):
            per_class_pred[row["cls"]].append(
                {"image_id": image_id, "box": row["box"], "conf": row["conf"]}
            )

    report = {}
    aps = []
    for cls_id, cls_name in enumerate(class_names):
        gts = per_class_gt[cls_id]
        if not gts and not per_class_pred[cls_id]:
            continue  # class absent from this eval split entirely
        result = evaluate_class(gts, per_class_pred[cls_id], iou_thres=iou_thres)
        report[cls_name] = result
        aps.append(result["ap"])

    report["_summary"] = {
        "mAP50": float(np.mean(aps)) if aps else 0.0,
        "n_images": len(image_ids),
        "n_classes_evaluated": len(aps),
    }
    return report
