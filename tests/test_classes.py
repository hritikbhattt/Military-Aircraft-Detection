import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from classes import CANONICAL_CLASSES, CLASS_TO_INDEX, INDEX_TO_CLASS, normalize_label
from prepare_dataset import voc_to_yolo


class TestClassNormalization(unittest.TestCase):
    def test_known_label_returns_unchanged(self):
        self.assertEqual(normalize_label("F16"), "F16")
        self.assertEqual(normalize_label("Su57"), "Su57")
        self.assertEqual(normalize_label("A10"), "A10")

    def test_whitespace_is_stripped(self):
        self.assertEqual(normalize_label(" F16 "), "F16")
        self.assertEqual(normalize_label("F16\n"), "F16")

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            normalize_label("Su27")
        with self.assertRaises(ValueError):
            normalize_label("Passenger")

    def test_exactly_103_classes(self):
        self.assertEqual(len(CANONICAL_CLASSES), 103)

    def test_no_duplicate_class_names(self):
        self.assertEqual(len(CANONICAL_CLASSES), len(set(CANONICAL_CLASSES)))

    def test_class_to_index_roundtrip(self):
        for i, name in enumerate(CANONICAL_CLASSES):
            self.assertEqual(CLASS_TO_INDEX[name], i)
            self.assertEqual(INDEX_TO_CLASS[i], name)


class TestVocToYolo(unittest.TestCase):
    def test_full_image_box(self):
        cx, cy, w, h = voc_to_yolo(0, 0, 100, 100, img_w=100, img_h=100)
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)
        self.assertAlmostEqual(w, 1.0)
        self.assertAlmostEqual(h, 1.0)

    def test_corner_box(self):
        # 10x10 box in the top-left of a 200x100 image
        cx, cy, w, h = voc_to_yolo(0, 0, 10, 10, img_w=200, img_h=100)
        self.assertAlmostEqual(cx, 10 / 200 / 2)
        self.assertAlmostEqual(cy, 10 / 100 / 2)
        self.assertAlmostEqual(w, 10 / 200)
        self.assertAlmostEqual(h, 10 / 100)

    def test_output_is_normalized_0_to_1(self):
        cx, cy, w, h = voc_to_yolo(50, 60, 150, 180, img_w=300, img_h=200)
        for v in (cx, cy, w, h):
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main()