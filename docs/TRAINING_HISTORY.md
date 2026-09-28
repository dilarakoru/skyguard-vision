# Training history

This document summarizes historical development experiments associated with SkyGuard Vision. It is intentionally separate from the current application checkpoint and should not be read as a benchmark for the model served by `detector.py`.

## Compute environments

Historical training notebooks record CUDA-enabled Ultralytics/PyTorch runs on:

- **NVIDIA A100-SXM4-40GB (40,507 MiB)** with PyTorch 2.6.0 + CUDA 12.4.
- **Tesla T4 (15,102 MiB)** with PyTorch 2.5.1 + CUDA 12.1.

The project also used Raspberry Pi hardware for edge-inference testing. The current repository does not package a complete Raspberry Pi deployment image or production edge stack.

## Model experimentation

Historical project artifacts include experiments with:

- YOLO11n
- YOLO11m
- YOLO11l
- YOLOv12L

Training records include 640-pixel image size runs, mixed-precision training, transfer/retraining experiments and saved Ultralytics artifacts such as `results.csv`, confusion matrices, PR/F1 curves and model weights.

A separate training script from the project history used 100 epochs, batch size 8, SGD, warmup, augmentation controls, resume training and `device=0` for GPU execution.

## Dataset quality work

Historical notebooks and reports include checks for:

- missing label files;
- corrupted images;
- image/annotation consistency;
- training/validation/test dataset preparation.

These checks were part of preparing the fire/smoke detection data for model experimentation.

## Example historical run

One recorded **YOLOv12L** experiment completed 100 epochs with the following final-epoch validation metrics:

| Metric | Value |
|---|---:|
| Precision | 0.8843 |
| Recall | 0.8242 |
| mAP@0.50 | 0.9009 |
| mAP@0.50:0.95 | 0.6369 |

These values belong to that specific historical training run and are **not** claimed as the performance of the checkpoint currently served by the application. Reproducing or publishing a model benchmark requires the exact dataset/split manifest, checkpoint linkage and independent evaluation.

## Current repository scope

The public application focuses on a small, reproducible inference path:

- local compatible checkpoint;
- CPU image inference;
- FastAPI/browser workflow;
- input validation;
- CLI export;
- automated API tests.

Historical training artifacts remain separate so the current application can stay lightweight while the project history remains documented.