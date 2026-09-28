"""Validate a YOLO fire/smoke dataset before training.

The historical notebooks included corrupted-image and annotation checks. This
standalone version keeps those checks local, deterministic and credential-free.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitReport:
    split: str
    images: int = 0
    labels: int = 0
    corrupted_images: int = 0
    missing_labels: int = 0
    orphan_labels: int = 0
    invalid_label_rows: int = 0


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.txt"))


def is_corrupted(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return False
    except Exception:
        return True


def invalid_yolo_rows(path: Path) -> int:
    invalid = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            invalid += 1
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, width, height = map(float, parts[1:])
        except ValueError:
            invalid += 1
            continue
        if class_id < 0 or not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            invalid += 1
    return invalid


def validate_split(root: Path, split: str) -> SplitReport:
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    images = image_files(images_dir)
    labels = label_files(labels_dir)

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    return SplitReport(
        split=split,
        images=len(images),
        labels=len(labels),
        corrupted_images=sum(is_corrupted(path) for path in images),
        missing_labels=len(image_stems - label_stems),
        orphan_labels=len(label_stems - image_stems),
        invalid_label_rows=sum(invalid_yolo_rows(path) for path in labels),
    )


def write_csv(reports: Iterable[SplitReport], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(report) for report in reports]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SplitReport.__annotations__))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check image/label integrity for YOLO train/valid/test splits.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--output", type=Path, default=Path("dataset_validation.csv"))
    args = parser.parse_args()

    reports = [validate_split(args.dataset_root, split) for split in args.splits]
    write_csv(reports, args.output)
    for report in reports:
        print(asdict(report))


if __name__ == "__main__":
    main()
