import tempfile
import unittest
from pathlib import Path

from research.dataset_validation import invalid_yolo_rows, validate_split


class DatasetValidationTests(unittest.TestCase):
    def test_invalid_rows_and_pair_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "train" / "images"
            labels = root / "train" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)

            from PIL import Image
            Image.new("RGB", (20, 20)).save(images / "ok.jpg")
            Image.new("RGB", (20, 20)).save(images / "missing.jpg")
            (labels / "ok.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            (labels / "orphan.txt").write_text("1.5 0.5 0.5 0 0.2\n", encoding="utf-8")

            self.assertEqual(invalid_yolo_rows(labels / "ok.txt"), 0)
            self.assertEqual(invalid_yolo_rows(labels / "orphan.txt"), 1)

            report = validate_split(root, "train")
            self.assertEqual(report.images, 2)
            self.assertEqual(report.labels, 2)
            self.assertEqual(report.missing_labels, 1)
            self.assertEqual(report.orphan_labels, 1)
            self.assertEqual(report.invalid_label_rows, 1)


if __name__ == "__main__":
    unittest.main()
