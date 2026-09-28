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

Recorded runs used combinations of 100 epochs, SGD, warmup, geometric/color augmentation and `device=0` on CUDA hardware. Exact batch size, learning rate and resume behavior varied by run; the saved `args.yaml` should be treated as the source of truth for any specific result.

## Dataset quality work

Historical notebooks and reports include checks for:

- missing label files;
- corrupted images;
- image/annotation consistency;
- training/validation/test dataset preparation.

The cleaned public utility in [../research/dataset_validation.py](../research/dataset_validation.py) preserves those checks without publishing private data or credentials.

## Example historical runs

### YOLO11L run on a fire/smoke dataset path

A recorded YOLO11L run points to `/content/fire-and-smoke-detection-2/data.yaml` and used 100 epochs, batch size 8, image size 640, SGD, AMP and CUDA device 0. Its final recorded validation row was:

| Metric | Value |
|---|---:|
| Precision | 0.6134 |
| Recall | 0.5901 |
| mAP@0.50 | 0.5994 |
| mAP@0.50:0.95 | 0.2584 |

The underlying dataset is not republished here, so these values are historical experiment evidence rather than a reproducible public benchmark.

### YOLOv12L run on the J.A.M. four-class dataset

A different recorded YOLOv12L run used the historical **J.A.M. Busqueda – Prototype 1** dataset. Its YAML names **fire, human, object and vehicle**; it is therefore not a fire/smoke-only benchmark. The final recorded validation row was:

| Metric | Value |
|---|---:|
| Precision | 0.8843 |
| Recall | 0.8242 |
| mAP@0.50 | 0.9009 |
| mAP@0.50:0.95 | 0.6369 |

These metrics belong to that specific four-class historical run and must not be attributed to the served fire/smoke checkpoint.

## Current repository scope

The public application focuses on a small, reproducible inference path:

- local compatible checkpoint;
- CPU image inference;
- FastAPI/browser workflow;
- input validation;
- CLI export;
- automated API tests.

Historical training artifacts remain separate so the current application can stay lightweight while the project history remains documented.
