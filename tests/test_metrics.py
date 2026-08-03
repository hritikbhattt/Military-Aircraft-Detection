import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

import numpy as np
from metrics import compute_ap, evaluate_class, evaluate_dataset, iou_xyxy, xywhn_to_xyxy


class TestIoU(unittest.TestCase):
    def test_identical_boxes_iou_is_1(self):
        box = np.array([0.1, 0.1, 0.5, 0.5])
        self.assertAlmostEqual(iou_xyxy(box, box), 1.0)

    def test_disjoint_boxes_iou_is_0(self):
        a = np.array([0.0, 0.0, 0.2, 0.2])
        b = np.array([0.5, 0.5, 0.7, 0.7])
        self.assertAlmostEqual(iou_xyxy(a, b), 0.0)

    def test_known_partial_overlap(self):
        # two unit-ish squares overlapping in a 1x1 region out of union 3x1... use simple numbers
        a = np.array([0.0, 0.0, 2.0, 1.0])  # area 2
        b = np.array([1.0, 0.0, 3.0, 1.0])  # area 2, overlap [1,0]-[2,1] area 1
        # union = 2+2-1 = 3, iou = 1/3
        self.assertAlmostEqual(iou_xyxy(a, b), 1 / 3, places=5)

    def test_xywhn_to_xyxy_conversion(self):
        box = np.array([0.5, 0.5, 0.4, 0.2])  # center 0.5,0.5 w=0.4 h=0.2
        x1, y1, x2, y2 = xywhn_to_xyxy(box)
        self.assertAlmostEqual(x1, 0.3)
        self.assertAlmostEqual(x2, 0.7)
        self.assertAlmostEqual(y1, 0.4)
        self.assertAlmostEqual(y2, 0.6)


class TestAP(unittest.TestCase):
    def test_perfect_detector_ap_is_1(self):
        recall = np.array([0.25, 0.5, 0.75, 1.0])
        precision = np.array([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(compute_ap(recall, precision), 1.0, places=3)

    def test_empty_recall_gives_zero(self):
        self.assertEqual(compute_ap(np.array([]), np.array([])), 0.0)

    def test_evaluate_class_perfect_match(self):
        gts = [{"image_id": "img1", "box": xywhn_to_xyxy(np.array([0.5, 0.5, 0.2, 0.2]))}]
        preds = [{"image_id": "img1", "box": xywhn_to_xyxy(np.array([0.5, 0.5, 0.2, 0.2])), "conf": 0.9}]
        result = evaluate_class(gts, preds, iou_thres=0.5)
        self.assertAlmostEqual(result["ap"], 1.0, places=3)
        self.assertEqual(result["n_gt"], 1)

    def test_evaluate_class_no_predictions_gives_zero_ap(self):
        gts = [{"image_id": "img1", "box": xywhn_to_xyxy(np.array([0.5, 0.5, 0.2, 0.2]))}]
        result = evaluate_class(gts, [], iou_thres=0.5)
        self.assertEqual(result["ap"], 0.0)
        self.assertEqual(result["recall"], 0.0)

    def test_evaluate_class_false_positive_hurts_precision(self):
        gts = [{"image_id": "img1", "box": xywhn_to_xyxy(np.array([0.5, 0.5, 0.2, 0.2]))}]
        preds = [
            {"image_id": "img1", "box": xywhn_to_xyxy(np.array([0.5, 0.5, 0.2, 0.2])), "conf": 0.9},
            {"image_id": "img1", "box": xywhn_to_xyxy(np.array([0.9, 0.9, 0.1, 0.1])), "conf": 0.8},
        ]
        result = evaluate_class(gts, preds, iou_thres=0.5)
        self.assertEqual(result["n_pred"], 2)
        self.assertLess(result["precision"], 1.0)


class TestEvaluateDataset(unittest.TestCase):
    def test_end_to_end_on_temp_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            gt_dir, pred_dir = Path(tmp) / "gt", Path(tmp) / "pred"
            gt_dir.mkdir(); pred_dir.mkdir()
            (gt_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n")
            (pred_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.2 0.95\n1 0.2 0.2 0.1 0.1 0.80\n")

            report = evaluate_dataset(str(gt_dir), str(pred_dir), ["F-22", "F-35"], iou_thres=0.5)
            self.assertAlmostEqual(report["_summary"]["mAP50"], 1.0, places=3)
            self.assertEqual(report["_summary"]["n_images"], 1)
            self.assertIn("F-22", report)
            self.assertIn("F-35", report)


if __name__ == "__main__":
    unittest.main()
