"""Historical-style CUDA training entry point for SkyGuard experiments.

This script is a cleaned, configurable version of the GPU training setup used in
earlier project notebooks. It intentionally does not download datasets or embed
credentials. Provide a local YOLO dataset YAML and local starting weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configurable Ultralytics YOLO GPU training experiment.")
    parser.add_argument("--data", type=Path, required=True, help="Local YOLO dataset YAML.")
    parser.add_argument("--weights", type=Path, required=True, help="Local Ultralytics-compatible starting weights.")
    parser.add_argument("--project", type=Path, default=Path("runs/research"), help="Directory for Ultralytics outputs.")
    parser.add_argument("--name", default="gpu_training", help="Ultralytics run name.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or 'cpu'.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", action="store_true", help="Cache images when memory allows.")
    parser.add_argument("--rect", action="store_true", help="Use rectangular batches.")
    parser.add_argument("--resume", action="store_true", help="Resume from the supplied checkpoint when supported.")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Starting weights not found: {args.weights}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        optimizer="SGD",
        warmup_epochs=3,
        patience=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.5,
        shear=2,
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        cache=args.cache,
        rect=args.rect,
        resume=args.resume,
        amp=True,
        seed=0,
        deterministic=True,
        device=args.device,
    )


if __name__ == "__main__":
    train(parse_args())
