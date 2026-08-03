import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from classes import CLASSES, class_index, normalize_raw_label
from prepare_dataset import voc_to_yolo


class TestClassNormalization(unittest.TestCase):
    def test_exact_codes_map_correctly(self):
        self.assertEqual(normalize_raw_label("F16"), "F-16")
        self.assertEqual(normalize_raw_label("SU57"), "Su-57")
        self.assertEqual(normalize_raw_label("A10"), "A-10")

    def test_case_and_punctuation_insensitive(self):
        self.assertEqual(normalize_raw_label("f-16"), "F-16")
        self.assertEqual(normalize_raw_label("f_16"), "F-16")
        self.assertEqual(normalize_raw_label(" F16 "), "F-16")

    def test_unsupported_label_returns_none(self):
        self.assertIsNone(normalize_raw_label("Su27"))
        self.assertIsNone(normalize_raw_label("Passenger"))

    def test_all_36_classes_have_at_least_one_raw_mapping(self):
        from classes import RAW_TO_CLASS
        mapped_targets = set(RAW_TO_CLASS.values())
        missing = set(CLASSES) - mapped_targets
        self.assertEqual(missing, set(), f"Classes with no raw-label mapping: {missing}")

    def test_no_duplicate_class_names(self):
        self.assertEqual(len(CLASSES), len(set(CLASSES)))

    def test_class_index_roundtrip(self):
        for i, name in enumerate(CLASSES):
            self.assertEqual(class_index(name), i)


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
